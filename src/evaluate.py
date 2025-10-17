import copy
import math
import os
import random
from pathlib import Path

import argparse
import pickle
import time

import torch
from torch_geometric.data import Data, Batch
import matplotlib.pyplot as plt
from shapely.geometry import Polygon as ShapelyPolygon

from tensorboardX import SummaryWriter

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .score_model import PolygonPackingTransformer
from .geometry_utils import Polygon as PolygonGeometry
from .geometry_utils import setRandomSeed
from .sde import init_sde, pc_sampler_state
from . import calutil
from . import rmspacing


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "datasets"
POLYGON_DIR = DATA_DIR / "polygons"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"

def cal_util(polys, pidsAll, actionsAll, eps=5.0):
    translationsAll = []
    thetasAll = []
    
    for i in range(len(pidsAll)):
        thetas = []
        translations = []
        for j in range(len(pidsAll[i])):
            thetas.append(float(math.atan2(actionsAll[i][j][3], actionsAll[i][j][2])))
            translations.append([float(actionsAll[i][j][0]), float(actionsAll[i][j][1])])
        thetasAll.append(thetas)
        translationsAll.append(translations)

    res = calutil.cal_util_all(pidsAll, thetasAll, copy.deepcopy(translationsAll), copy.deepcopy(polys), eps)

    # sumarea, maxinter, suminter, minx, miny, maxx, maxy
    res = torch.FloatTensor(res)
    
    bbd_area = (res[:, 5] - res[:, 3]) * (res[:, 6] - res[:, 4])
    util = res[:, 0] / bbd_area
    valid = (res[:, 1] < eps)
    bbd = torch.stack([res[:, 3], res[:, 4], res[:, 5], res[:, 6]], dim=1)
    sum_inter_per = res[:, 2] / res[:, 0] * 100
    
    return util, valid, bbd, sum_inter_per

def existsOrMkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)

def readAllPolys(polyCnt=440):
    polyVertices = []
    for i in range(0, polyCnt):
        # print(i)
        poly_path = POLYGON_DIR / f"{i}.txt"
        poly = PolygonGeometry(str(poly_path))
        maxContour = poly.getMaxContour()
        polyVertices.append(maxContour)
    
    return polyVertices

def genPaddingMask(polyVerticesData):
    paddingMaskData = []
    for i in range(len(polyVerticesData)):
        paddingMask = []
        for j in range(len(polyVerticesData[i])):
            paddingMask.append(0)
        #for j in range(len(polyVerticesData[i]), 50):
        #    paddingMask.append(-inf)
        paddingMaskData.append(paddingMask)
    return paddingMaskData

def padData(polyIds, actions):
    for i in range(len(polyIds)):
        for j in range(len(actions[i])):
            actions[i][j][2] *= 500
            actions[i][j][3] *= 500
        #for j in range(len(polyIds[i]), 50):
        #    polyIds[i].append(0)
        #    actions[i].append([0, 0, 0, 0])
    return polyIds, actions

            

def collectValiData():
    
    dirName = DATASET_DIR / "dataset_dental_vali_128.pkl"
    dataFile = open(dirName, "rb")
    print("loading data from ", dirName)
    polyIds = pickle.load(dataFile)
    actions = pickle.load(dataFile)
    paddingMask = genPaddingMask(polyIds)
    polyIds, actions = padData(polyIds, actions)
    
    dataFile.close()

    
    return polyIds, actions, paddingMask


def removeSpacing(polys, pidsAll, actionsAll, height):
    translationsAll = []
    thetasAll = []
    
    for i in range(len(pidsAll)):
        thetas = []
        translations = []
        for j in range(len(pidsAll[i])):
            thetas.append(float(math.atan2(actionsAll[i][j][3], actionsAll[i][j][2])))
            translations.append([float(actionsAll[i][j][0]), float(actionsAll[i][j][1])])
        thetasAll.append(thetas)
        translationsAll.append(translations)

    rm_begin = time.time()
    rm_translations = rmspacing.rm_spacing_all(pidsAll, thetasAll, copy.deepcopy(translationsAll), copy.deepcopy(polys), 8000.0, height, 1.0)
    rm_end = time.time()
    
    newActionsAll = []
    for i in range(len(pidsAll)):
        newActions = []
        for j in range(len(pidsAll[i])):
            newActions.append([float(rm_translations[i][j][0]), float(rm_translations[i][j][1]), float(actionsAll[i][j][2]), float(actionsAll[i][j][3])])
        newActionsAll.append(newActions)
    
    return torch.FloatTensor(newActionsAll), rm_end - rm_begin


def calculate_angle(p1, p2, p3):
    a = [p1[0] - p2[0], p1[1] - p2[1]]
    b = [p3[0] - p2[0], p3[1] - p2[1]]
    dot_product = a[0] * b[0] + a[1] * b[1]
    cross_product = a[0] * b[1] - a[1] * b[0]
    angle = math.atan2(cross_product, dot_product)
    angle = abs(angle) * (180.0 / math.pi)
    if cross_product < 0:
        angle = 360 - angle
    return angle

def compute_node_features(poly):
    node_features = []
    
    for i in range(len(poly)):
        p1, p2, p3 = poly[i - 1], poly[i], poly[(i + 1) % len(poly)]
        # 计算内角
        internal_angle = calculate_angle(p1, p2, p3)
        node_features.append([p2[0], p2[1], internal_angle])

    return node_features

def compute_global_features(poly):
    polygon = ShapelyPolygon(poly)
    area = polygon.area
    perimeter = polygon.length

    return [area, perimeter]
        
def create_gnn_data(polygons):
    cum_node_count = 0 
    
    data_list = []
    
    for poly_index, poly in enumerate(polygons):
        node_features = compute_node_features(poly)  # shape: (n, 3)
        area, perm = compute_global_features(poly) # shape (2)

        num_nodes = len(poly)
        edge_indices = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
        edge_indices += [((i + 1) % num_nodes, i) for i in range(num_nodes)]
        # edge_indices = [(u + cum_node_count, v + cum_node_count) for u, v in edge_indices]

        cum_node_count += num_nodes

        node_features_tensor = torch.tensor(node_features, dtype=torch.float)
        edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        area_tensor = torch.tensor(area, dtype=torch.float)
        perm_tensor = torch.tensor(perm, dtype=torch.float)
            
        data = Data(x=node_features_tensor, edge_index=edge_index_tensor, area=area_tensor, perm=perm_tensor)
        data_list.append(data)

    # print(cum_node_count)
    batched_data = Batch.from_data_list(data_list)

    return batched_data

def rotatePoly(poly, theta):
    # poly shape is 1, n, 2
    # poly = poly.unsqueeze(0)
    poly = poly.transpose(1, 2)

    cosTheta = torch.cos(theta)
    sinTheta = torch.sin(theta)
    rotationMatrices = torch.stack([cosTheta, -sinTheta, sinTheta, cosTheta], dim=1).view(-1, 2, 2)

    # print(rotationMatrices.shape, (poly - centers).shape)
    rotatedPoly = torch.bmm(rotationMatrices, poly)
    
    return rotatedPoly.squeeze(0).transpose(0, 1)

def visualize(epoch, polyIds, polyVertices, predict, paddingMask, writer, figName):
    predict = predict.cpu()
    
    plt.figure()
    for i, polyId in enumerate(polyIds):
        if paddingMask[i] < 0: continue
        polyVertex = polyVertices[polyId]
        polyTensor = torch.FloatTensor([polyVertex]).clone()

        theta = torch.atan2(predict[i, 3], predict[i, 2]).unsqueeze(0)
        rotatedPoly = rotatePoly(polyTensor, theta)
        
        res = torch.FloatTensor([[predict[i, 0], predict[i, 1]]])
        newPoly = rotatedPoly + res
        newPoly = torch.cat((newPoly, newPoly[0].unsqueeze(0)), dim=0)
        # print(newPoly)
        plt.plot([float(p[0]) for p in newPoly], [float(p[1]) for p in newPoly])

    # plt.xlim(0, 2560)
    # plt.ylim(0, 7000)
    plt.title(figName + '_epoch_{}'.format(epoch))
    writer.add_figure(f"Images/epoch_{epoch}_{figName}", plt.gcf())
    plt.clf()
    plt.close()


def vali_res(valiPolyIds, polyVertices, valiActions, paddingMaskData, predict):
    
    polyList_toc = []
    polyIds_toc = []
    polyCnt = 0

    for i, polyId in enumerate(valiPolyIds):
        if paddingMaskData[i] < 0:
            continue
        polyList_toc.append(copy.deepcopy(polyVertices[polyId]))
        polyIds_toc.append(polyCnt)
        polyCnt += 1
    
        polyIdsAll_toc = []
        actionsAll_toc = []
        for i in range(len(predict)):
            polyIdsAll_toc.append(copy.deepcopy(polyIds_toc))
            actionsAll_toc.append(copy.deepcopy(predict[i, :polyCnt].cpu().tolist()))
        
        polyIdsAll_toc.append(copy.deepcopy(polyIds_toc))
        actionsAll_toc.append(copy.deepcopy(valiActions[:polyCnt].cpu().tolist()))
        
    util, valid, bbd, sum_inter_per = cal_util(copy.deepcopy(polyList_toc), polyIdsAll_toc, actionsAll_toc, 50.0)
    
    before_vali_cnt = 0
    before_vali_util_list = []
    before_intersec_area_list = []
    for i in range(len(util) - 1):
        before_intersec_area_list.append(sum_inter_per[i])
        if valid[i]:
            before_vali_cnt += 1
            before_vali_util_list.append(util[i])
    
    for i in range(len(predict)):
        predict[i, :polyCnt, 0] -= bbd[i, 0]
        predict[i, :polyCnt, 1] -= bbd[i, 1]
        actionsAll_toc[i] = predict[i, :polyCnt].cpu().tolist()
    
    rm_predict, rm_time = removeSpacing(copy.deepcopy(polyList_toc), polyIdsAll_toc, actionsAll_toc, 1205.0)
    # rm_predict = torch.FloatTensor(actionsAll_toc)

    rm_actionsAll_toc = []
    # include ground truth actions
    for i in range(len(actionsAll_toc)):
        rm_actionsAll_toc.append(rm_predict[i, :polyCnt].cpu().tolist())
        
    rm_util, rm_valid, rm_bbd, rm_sum_inter_per = cal_util(copy.deepcopy(polyList_toc), polyIdsAll_toc, rm_actionsAll_toc, 50.0)
    
    intersec_area_list = []
    rm_vali_util_list = []
    
    rm_vali_cnt = 0
    
    best_util = 0
    best_id = 0
    for i in range(len(rm_util) - 1):
        intersec_area_list.append(rm_sum_inter_per[i])
        if rm_valid[i]:
            rm_vali_cnt += 1
            rm_vali_util_list.append(rm_util[i])
            if rm_util[i] > best_util:
                best_util = rm_util[i]
                best_id = i
                
    gt_util = util[-1]
    
    if len(rm_vali_util_list) == 0:
        rm_vali_util_list.append(0)
    if len(before_vali_util_list) == 0:
        before_vali_util_list.append(0)    
    
    return before_vali_cnt, rm_vali_cnt, gt_util, before_intersec_area_list, intersec_area_list, before_vali_util_list, rm_vali_util_list, rm_predict, best_id, rm_predict[-1], rm_time

if __name__ == '__main__':
    print("hi")
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='test')
    parser.add_argument('--n', type=int, default=192)
    parser.add_argument('--m', type=int, default=192)
    parser.add_argument('--x', type=int, default=2000)
    parser.add_argument('--y', type=int, default=2000)
    
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--beginEpoch', type=int, default=0)
    parser.add_argument('--grad_clip', type=float, default=1.)
    parser.add_argument('--ema_rate', type=float, default=0.999)
    parser.add_argument('--repeat_num', type=int, default=1)
    parser.add_argument('--sigma', type=float, default=25.)
    parser.add_argument('--sde_mode', type=str, default='ve')
    
    parser.add_argument('--n_epochs', type=int, default=1000000)
    parser.add_argument('--visualize_freq', type=int, default=256)
    parser.add_argument('--vali_freq', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--seed', type=int, default=3407)
    # load args
    
    args = parser.parse_args()
    batchSize = args.batch_size
    beginEpoch = args.beginEpoch
    # n, m, x, y = args.n, args.m, args.x, args.y
    
    
    # use torch.distributed.launch to set the local rank automatically
    localRank = int(os.environ["LOCAL_RANK"])
    worldSize = int(os.environ['WORLD_SIZE'])
    # initialize the process group
    print("preparing process group...")
    dist.init_process_group("nccl", rank=localRank, world_size=worldSize)
    
    setRandomSeed(args.seed + localRank)
    
    # create the model and move it to the local GPU
    device = torch.device("cuda", localRank)
    print("device locked ", localRank)
    torch.cuda.set_device(localRank)
    torch.cuda.empty_cache()
    torch.set_num_threads(16)
    
    if localRank == 0:
        existsOrMkdir(LOG_DIR)
        tb_path = LOG_DIR / args.log_dir / "eval"
        existsOrMkdir(tb_path)
        writer = SummaryWriter(str(tb_path), "base")
    
    print("reading polys...")
    polyVertices = readAllPolys(440)
    
    gnnFeatureData = create_gnn_data(polyVertices)
    gnnFeatureData = gnnFeatureData.to(device)
    
    polyVertexNumbers = []
    for polyVertex in polyVertices:
        polyVertexNumbers.append(len(polyVertex))
        
    polyIdsVali, actionsVali, paddingMaskVali = collectValiData()
    polyIdsVali = polyIdsVali[:128]
    actionsVali = actionsVali[:128]
    paddingMaskVali = paddingMaskVali[:128]

    prior_fn, marginal_prob_fn, sde_fn, sampling_eps = init_sde(args.sde_mode)
    
    # print(edges)
    ''' Init Model '''
    
    score = PolygonPackingTransformer(marginal_prob_std_func=marginal_prob_fn, device=device).to(device)
    checkpoint_path = CHECKPOINT_DIR / "score_model.pth"
    if checkpoint_path.exists():
        scoreState = torch.load(checkpoint_path, map_location="cpu")
        score.load_state_dict(scoreState)
        if localRank == 0:
            print(f"Loaded pretrained weights from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Required checkpoint not found at {checkpoint_path}")

    score = DDP(score, device_ids=[localRank], output_device=localRank, find_unused_parameters=False)
    paramToLearn = filter(lambda p: p.requires_grad, score.module.parameters())
    # create the dataset and the distributed sampler
    
    
    epoch = 0
    
    if localRank == 0:
        choosedVali = random.randint(0, 128)
        valiPolyIds = torch.tensor(polyIdsVali[choosedVali],dtype=torch.int64).squeeze(0)
        valiActions = torch.tensor(actionsVali[choosedVali],dtype=torch.float32).squeeze(0)
        paddingMaskData = torch.tensor(paddingMaskVali[choosedVali],dtype=torch.float32).squeeze(0)
        # valiHeights = torch.tensor(heightsDataAll[choosedVali],dtype=torch.float32).squeeze(0)
    
    
        with torch.no_grad():
            samples, res = pc_sampler_state(score.module, sde_fn, len(valiPolyIds), valiPolyIds, gnnFeatureData, paddingMaskData)
        
        before_vali_cnt, rm_vali_cnt, gt_util, \
        before_intersec_area_list, intersec_area_list, \
        before_vali_util_list, rm_vali_util_list, \
        rm_predict, best_util_id, rm_vali_action,\
        rm_time = vali_res(valiPolyIds, polyVertices, valiActions, paddingMaskData, res)
        
        visualize(epoch, valiPolyIds, polyVertices, rm_vali_action, paddingMaskData, writer, "gt")
        visualize(epoch, valiPolyIds, polyVertices, rm_predict[best_util_id], paddingMaskData, writer, "pr")
        
        rm_best_util = max(rm_vali_util_list)
        rm_avg_util = sum(rm_vali_util_list) / len(rm_vali_util_list)
        rm_wrst_util = min(rm_vali_util_list)
        
        before_best_util = max(before_vali_util_list)
        before_avg_util = sum(before_vali_util_list) / len(before_vali_util_list)
        before_wrst_util = min(before_vali_util_list)
        
        before_best_inter = min(before_intersec_area_list)
        before_avg_inter = sum(before_intersec_area_list) / len(before_intersec_area_list)
        before_wrst_inter = max(before_intersec_area_list)
        
        print("before_best_inter=%f before_avg_inter=%f before_wrst_inter=%f" % (before_best_inter, before_avg_inter, before_wrst_inter))
        print("rm_best_util=%f rm_avg_util=%f rm_wrst_util=%f" % (rm_best_util, rm_avg_util, rm_wrst_util))
        print("before_vali_cnt=%d rm_vali_cnt=%d" % (before_vali_cnt, rm_vali_cnt))
        
        writer.add_scalars('vali/util', {'PRBest': rm_best_util}, epoch)
        writer.add_scalars('vali/util', {'PRAvg': rm_avg_util}, epoch)
        writer.add_scalars('vali/util', {'PRWorst': rm_wrst_util}, epoch)
        
        writer.add_scalars('vali/area', {'PRBest': before_best_inter}, epoch)
        writer.add_scalars('vali/area', {'PRAvg': before_avg_inter}, epoch)
        writer.add_scalars('vali/area', {'PRWorst': before_wrst_inter}, epoch)
        
        writer.add_scalars('vali/util', {'GT': gt_util}, epoch)
        
        writer.add_scalars('vali/validCnt', {'bef': before_vali_cnt}, epoch)
        writer.add_scalars('vali/validCnt', {'aft': rm_vali_cnt}, epoch)
    
    if localRank == 0:
        writer.close()
    dist.destroy_process_group()
