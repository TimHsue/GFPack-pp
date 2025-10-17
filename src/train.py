import copy
import math
import os
import random
from pathlib import Path

import numpy as np
from tqdm import trange
import argparse
import pickle
import time

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import matplotlib.pyplot as plt
from shapely.geometry import Polygon as ShapelyPolygon

from tensorboardX import SummaryWriter

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .score_model import PolygonPackingTransformer
from .geometry_utils import Polygon as PolygonGeometry
from .geometry_utils import setRandomSeed
from .sde import init_sde, lossFun, pc_sampler_state, ExponentialMovingAverage
from . import calutil
from . import rmspacing


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "datasets"
POLYGON_DIR = DATA_DIR / "polygons"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
SNAPSHOT_DIR = CHECKPOINT_DIR / "snapshots"
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

def collectData():
    
    dirName = DATASET_DIR / "dataset_dental_sp7_r_h12_new.pkl"
    dataFile = open(dirName, "rb")
    print("load data from ", dirName)
    polyIds = pickle.load(dataFile)
    actions = pickle.load(dataFile)
    paddingMask = genPaddingMask(polyIds)
    polyIds, actions = padData(polyIds, actions)
    
    dataFile.close()
    #print(polyIds.shape)
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

def collectValiData():
    
    dirName = DATASET_DIR / "dataset_dental_vali_128.pkl"
    dataFile = open(dirName, "rb")
    print("load data from ", dirName)
    polyIds = pickle.load(dataFile)
    actions = pickle.load(dataFile)
    paddingMask = genPaddingMask(polyIds)
    polyIds, actions = padData(polyIds, actions)
    
    dataFile.close()

    
    return polyIds, actions, paddingMask

def calculate_angle(p1, p2, p3):
    """计算由点 p1, p2, p3 形成的角的大小（p2 是顶点），适用于非凸多边形。"""
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


def cal_weight_dataset(polys, pidsAll, actionsAll, eps=5.0):
    util, valid, bbd, sum_inter_per = cal_util(copy.deepcopy(polys), pidsAll, actionsAll, 50.0)
    min_util = util.min()
    max_util = util.max()
    avg_util = util.mean()
    
    print(min_util, max_util, avg_util)
    # weight =  * 10
    # weight = torch.softmax((util - min_util) / (max_util - min_util) * 10.0, dim=0)
    weight = torch.sigmoid((util - avg_util) / (max_util - min_util) * 10.0)
    return weight.tolist()

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

def vali_all(id_list, action_list, padding_list, score, sde_fn, gnnFeatureData, polyVertices):
    
    before_sum_inter_all = 0
    before_wrst_inter_all = 0
    
    before_vali_cnt_all = 0
    rm_vali_cnt_all = 0
    
    rm_wrst_util_all = 1000
    rm_sum_util_all = 0
    rm_best_util_all = 0
    
    total_gen_time = 0
    total_rm_time = 0
    
    bef_wrst_util_all = 1000
    bef_sum_util_all = 0
    bef_best_util_all = 0
    
    for choosedVali in trange(len(id_list)):
        valiPolyIds = torch.tensor(id_list[choosedVali],dtype=torch.int64).squeeze(0)
        valiActions = torch.tensor(action_list[choosedVali],dtype=torch.float32).squeeze(0)
        paddingMaskData = torch.tensor(padding_list[choosedVali],dtype=torch.float32).squeeze(0)
        
        with torch.no_grad():
            gen_time_begin = time.time()
            samples, res = pc_sampler_state(score, sde_fn, len(valiPolyIds), valiPolyIds, gnnFeatureData, paddingMaskData)
            gen_time_end = time.time()
        total_gen_time += gen_time_end - gen_time_begin
        
        before_vali_cnt, rm_vali_cnt, gt_util, \
        before_intersec_area_list, rm_intersec_area_list, \
        before_vali_util_list, rm_vali_util_list, \
        rm_predict, best_util_id, rm_vali_actions, \
        rm_time = vali_res(valiPolyIds, polyVertices, valiActions, paddingMaskData, res)
        
        total_rm_time += rm_time
        
        before_vali_cnt_all += before_vali_cnt
        rm_vali_cnt_all += rm_vali_cnt
        
        before_avg_inter = sum(before_intersec_area_list) / len(before_intersec_area_list)
        before_wrst_inter = max(before_intersec_area_list)
        
        before_wrst_inter_all = max(before_wrst_inter_all, before_wrst_inter)
        before_sum_inter_all += before_avg_inter
        
        rm_best_util = max(rm_vali_util_list)
        bef_best_util = max(before_vali_util_list)
        
        rm_best_util_all = max(rm_best_util_all, rm_best_util)
        rm_sum_util_all += rm_best_util
        rm_wrst_util_all = min(rm_wrst_util_all, rm_best_util)
        
        bef_best_util_all = max(bef_best_util_all, bef_best_util)
        bef_sum_util_all += bef_best_util
        bef_wrst_util_all = min(bef_wrst_util_all, bef_best_util)
        
    return  int(rm_vali_cnt_all), float(rm_sum_util_all), float(rm_wrst_util_all), \
            float(rm_best_util_all), float(before_sum_inter_all), float(before_wrst_inter_all), int(before_vali_cnt_all),\
            float(bef_sum_util_all), float(bef_wrst_util_all), float(bef_best_util_all), \
            float(total_gen_time), float(total_rm_time)
    
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
        tb_path = LOG_DIR / args.log_dir / "train"
        existsOrMkdir(tb_path)
        existsOrMkdir(CHECKPOINT_DIR)
        existsOrMkdir(SNAPSHOT_DIR)
        writer = SummaryWriter(str(tb_path), "base")
    
    print("reading polys...")
    polyVertices = readAllPolys(440)
    gnnFeatureData = create_gnn_data(polyVertices)
    gnnFeatureData = gnnFeatureData.to(device)
    
    polyVertexNumbers = []
    for polyVertex in polyVertices:
        polyVertexNumbers.append(len(polyVertex))
        
    print("loading teacher...")
    polyIdsDataAll, actionsDataAll, paddingMaskDataAll = collectData()

    allDataSize = len(polyIdsDataAll)
    datasetSize = int(allDataSize * 1.0)
    print("datasetSize=%d allDataSize=%d" % (datasetSize, allDataSize))
    polyIdsData = polyIdsDataAll[:datasetSize]
    actionsData = actionsDataAll[:datasetSize]
    paddingMaskData = paddingMaskDataAll[:datasetSize]
    
    weightsAll = cal_weight_dataset(polyVertices, polyIdsData, actionsData)
    
    
    polyIdsVali, actionsVali, paddingMaskVali = collectValiData()
    polyIdsVali = polyIdsVali[:128]
    actionsVali = actionsVali[:128]
    paddingMaskVali = paddingMaskVali[:128]

    prior_fn, marginal_prob_fn, sde_fn, sampling_eps = init_sde(args.sde_mode)
    
    # Init Model
    
    score = PolygonPackingTransformer(marginal_prob_std_func=marginal_prob_fn, device=device).to(device)
    checkpoint_path = CHECKPOINT_DIR / "score_model.pth"
    checkpoint_pickle_path = CHECKPOINT_DIR / "score_model.pkl"
    if checkpoint_path.exists():
        scoreState = torch.load(checkpoint_path, map_location="cpu")
        score.load_state_dict(scoreState)
        if localRank == 0:
            print(f"Loaded pretrained weights from {checkpoint_path}")
    else:
        if localRank == 0:
            print(f"No pretrained checkpoint found at {checkpoint_path}, initializing from scratch.")

    score = DDP(score, device_ids=[localRank], output_device=localRank, find_unused_parameters=False)
    paramToLearn = filter(lambda p: p.requires_grad, score.module.parameters())
    optimizer = optim.AdamW(paramToLearn, lr=args.lr, weight_decay=1e-4)
    # create the dataset and the distributed sampler
    
    dataset = list(map(lambda x, y, z, w: 
        Data(x=torch.tensor(x, dtype=torch.float32), 
             y=torch.tensor(y, dtype=torch.int64), 
             z=torch.tensor(z, dtype=torch.float32),
             w=torch.tensor(w, dtype=torch.float32)),
        actionsData, polyIdsData, paddingMaskData, weightsAll))
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=False, num_workers=0, sampler=sampler)
    
    ema = ExponentialMovingAverage(score.parameters(), decay=args.ema_rate)
    # optimizer = optim.Adam(score.parameters(), lr=2e-4)
    
    numberEpochs = args.n_epochs
    iterPerEpoch = int(datasetSize // batchSize)
    print("Starting Training Loop...")
    
    bestCnt = 0
    
    curIndex = 0
    for epoch in trange(numberEpochs):
        # For each batch in the dataloader
        totLoss = 0
        totDelta = 0
        totLen = 0
        sampler.set_epoch(epoch)
        
        for i, choosedData in enumerate(dataloader):
            nowLoss = 0
            nowDelta = 0
            for _ in range(args.repeat_num):
                # calc score-matching loss
                loss, delta = lossFun(score, choosedData, gnnFeatureData, marginal_prob_fn)
                optimizer.zero_grad()
                loss.backward()
                
                nowDelta += delta.abs().mean()
                nowLoss += loss
                
            nowLoss /= args.repeat_num
            nowDelta /= args.repeat_num
            
            if localRank == 0:
                writer.add_scalars('train/train_loss',  {'current': nowLoss.item()}, curIndex)
                writer.add_scalar('train/train_delta', nowDelta.item(), curIndex)
        
            totLoss += nowLoss.item() * len(choosedData)
            totDelta += nowDelta.item() * len(choosedData)
            totLen += len(choosedData)
            
            if args.warmup > 0 and curIndex < args.warmup:
                for g in optimizer.param_groups:
                    g['lr'] = args.lr * np.minimum(curIndex / args.warmup, 1.0)
            
            # grad clip
            if args.grad_clip >= 0:
                torch.nn.utils.clip_grad_norm_(score.parameters(), max_norm=args.grad_clip)
            
            optimizer.step()
            
            ema.update(score.parameters())
            
            if args.ema_rate > 0 and curIndex % 8 == 0:
                ema.store(score.parameters())
                ema.copy_to(score.parameters())
                ema.restore(score.parameters())
            curIndex += 1
            
        print()
        print("Epoch: {}, Loss: {}, Delta: {}".format(epoch, totLoss / totLen, totDelta / totLen))
        
        if (epoch) % 2 == 0 and localRank == 0:
            torch.save(score, str(checkpoint_pickle_path))
            torch.save(score.module.state_dict(), str(checkpoint_path))
            
        if (epoch) % args.vali_freq == 0 and localRank == 0:
            snapshot_pickle = SNAPSHOT_DIR / f"score_model_epoch{epoch}.pkl"
            snapshot_weights = SNAPSHOT_DIR / f"score_model_epoch{epoch}.pth"
            torch.save(score, str(snapshot_pickle))
            torch.save(score.module.state_dict(), str(snapshot_weights))
        
        
        if (epoch) % args.vali_freq == 0:
            vali_whole_size = len(polyIdsVali)
            vali_per_gpu = vali_whole_size // worldSize
            vali_begin = vali_per_gpu * localRank
            vali_end = min(vali_per_gpu * (localRank + 1), vali_whole_size)
        
            vali_info = vali_all(polyIdsVali[vali_begin:vali_end], actionsVali[vali_begin:vali_end], paddingMaskVali[vali_begin:vali_end], score.module, sde_fn, gnnFeatureData, polyVertices)
            # vali_info: rm_vali_cnt_all, rm_sum_util_all, rm_wrst_util_all, before_sum_inter_all, before_wrst_inter_all, before_vali_cnt_all
            
            rm_vali_cnt_all = torch.tensor(vali_info[0], dtype=torch.int64).to(device)
            rm_sum_util_all = torch.tensor(vali_info[1], dtype=torch.float32).to(device)
            rm_wrst_util_all = torch.tensor(vali_info[2], dtype=torch.float32).to(device)
            rm_best_util_all = torch.tensor(vali_info[3], dtype=torch.float32).to(device)
            before_sum_inter_all = torch.tensor(vali_info[4], dtype=torch.float32).to(device)
            before_wrst_inter_all = torch.tensor(vali_info[5], dtype=torch.float32).to(device)
            before_vali_cnt_all = torch.tensor(vali_info[6], dtype=torch.int64).to(device)
            bef_sum_util_all = torch.tensor(vali_info[7], dtype=torch.float32).to(device)
            bef_wrst_util_all = torch.tensor(vali_info[8], dtype=torch.float32).to(device)
            bef_best_util_all = torch.tensor(vali_info[9], dtype=torch.float32).to(device)
            total_gen_time = torch.tensor(vali_info[10], dtype=torch.float32).to(device)
            total_rm_time = torch.tensor(vali_info[11], dtype=torch.float32).to(device)
            
            
            dist.all_reduce(rm_vali_cnt_all)
            dist.all_reduce(rm_sum_util_all)
            dist.all_reduce(rm_wrst_util_all, op=dist.ReduceOp.MIN)
            dist.all_reduce(rm_best_util_all, op=dist.ReduceOp.MAX)
            dist.all_reduce(before_sum_inter_all)
            dist.all_reduce(before_wrst_inter_all, op=dist.ReduceOp.MAX)
            dist.all_reduce(before_vali_cnt_all)
            dist.all_reduce(bef_sum_util_all)
            dist.all_reduce(bef_wrst_util_all, op=dist.ReduceOp.MIN)
            dist.all_reduce(bef_best_util_all, op=dist.ReduceOp.MAX)
            dist.all_reduce(total_gen_time)
            dist.all_reduce(total_rm_time)
            
            if localRank == 0:
                rm_vali_cnt_all = float(rm_vali_cnt_all.item()) / float(vali_whole_size)
                before_vali_cnt_all = float(before_vali_cnt_all.item()) / float(vali_whole_size)
                rm_sum_util_all = float(rm_sum_util_all.item()) / float(vali_whole_size)
                bef_sum_util_all = float(bef_sum_util_all.item()) / float(vali_whole_size)
                before_sum_inter_all = float(before_sum_inter_all.item()) / float(vali_whole_size)
                total_gen_time = float(total_gen_time.item()) / float(vali_whole_size)
                total_rm_time = float(total_rm_time.item()) / float(vali_whole_size)
                
                print("-------------------------------")
                print("before_vali_cnt_all=%d rm_vali_cnt_all=%d" % (before_vali_cnt_all, rm_vali_cnt_all))
                print("before_sum_inter_all=%f before_wrst_inter_all=%f" % (before_sum_inter_all, before_wrst_inter_all.item()))
                print("rm_sum_util_all=%f rm_wrst_util_all=%f rm_best_util_all=%f" % (rm_sum_util_all, rm_wrst_util_all.item(), rm_best_util_all.item()))
                print("bef_sum_util_all=%f bef_wrst_util_all=%f bef_best_util_all=%f" % (bef_sum_util_all, bef_wrst_util_all.item(), bef_best_util_all.item()))
                print("total_gen_time=%f total_rm_time=%f" % (total_gen_time, total_rm_time))
                print("-------------------------------")
                
                writer.add_scalars('valiall/validCnt', {'rm': rm_vali_cnt_all}, epoch)
                writer.add_scalars('valiall/util', {'rm': rm_sum_util_all}, epoch)
                writer.add_scalars('valiall/util', {'rmWrst': rm_wrst_util_all.item()}, epoch)
                writer.add_scalars('valiall/util', {'rmBest': rm_best_util_all.item()}, epoch)
                writer.add_scalars('valiall/area', {'bef': before_sum_inter_all}, epoch)
                writer.add_scalars('valiall/area', {'befWrst': before_wrst_inter_all.item()}, epoch)
                writer.add_scalars('valiall/validCnt', {'bef': before_vali_cnt_all}, epoch)
                writer.add_scalars('valiall/util', {'bef': bef_sum_util_all}, epoch)
                writer.add_scalars('valiall/util', {'befWrst': bef_wrst_util_all.item()}, epoch)
                writer.add_scalars('valiall/util', {'befBest': bef_best_util_all.item()}, epoch)
                writer.add_scalars('valiall/time', {'gen': total_gen_time}, epoch)
                writer.add_scalars('valiall/time', {'rm': total_rm_time}, epoch)
                
        
        if (epoch) % args.visualize_freq == 0 and localRank == 0:
            choosedVali = random.randint(0, allDataSize - 1)
            valiPolyIds = torch.tensor(polyIdsDataAll[choosedVali],dtype=torch.int64).squeeze(0)
            valiActions = torch.tensor(actionsDataAll[choosedVali],dtype=torch.float32).squeeze(0)
            paddingMaskData = torch.tensor(paddingMaskDataAll[choosedVali],dtype=torch.float32).squeeze(0)
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
