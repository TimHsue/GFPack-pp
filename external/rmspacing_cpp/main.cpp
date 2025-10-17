//#include "clipper.h"
#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "AABB.hpp"
#include "clipper.h"
#include <time.h>
#include <chrono>
#include <omp.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <utility>

namespace py = pybind11;


using std::cout;
using std::cin;
using std::endl;
using std::vector;
using std::pair;
using std::make_pair;
using std::min;
using std::max;
using aabb::Tree;
using namespace Clipper2Lib;
const double Pi=3.1415926535897932384626433832795;

pair<double,double> rotateP(const pair<double,double> p, double angle) {
    pair<double,double> result;

    result.first = p.first * cos(angle) - p.second * sin(angle);
    result.second = p.first * sin(angle) + p.second * cos(angle) ;

    return result;
}

pair<double,double> translateP(const pair<double,double> p, pair<double,double> q) {
    pair<double,double> result;

    result.first = p.first + q.first ;
    result.second = p.second + q.second  ;

    return result;
}
vector<pair<double,double>> translate(const vector<pair<double,double>> p, pair<double,double> q) {
    vector<pair<double,double>> result;
    for(auto a:p){
        result.push_back( translateP(a,q));
    }
    return result;
}
vector<pair<double,double>> readpolygon(int k){
    std::ostringstream oss;
    oss << "polys/"<<k<<".txt";

    std::ifstream inputFile(oss.str());
    //cout<<oss.str()<<endl;

    int n,p;
    double x,y;
    inputFile>>n;
    vector<pair<double,double>> points;
    for(int i=0;i<n;i++){
        inputFile>>p;
        for(int j=0;j<p;j++){
            inputFile>>x>>y;
            points.push_back(make_pair(10*x,10*y));
        }
    }
    inputFile.close();
    return points;
}
vector<vector<pair<double,double>>> open(int n){
    vector<vector<pair<double,double>>> re;
    for(int i=0;i<n;i++){
        vector<pair<double,double>> p =readpolygon(i);
        //cout<<p.size()<<endl;
        re.push_back(p);
    }
    return re;
}
void read2(int ep,vector<int>&ids,vector<pair<double,double>>& actions,vector<double>&rotates){
    
    std::ostringstream oss;
    oss <<"results/result_"<<ep<<".txt";
    
    //cout<<oss.str()<<endl;
    std::ifstream inputFile(oss.str());

    if (!inputFile.is_open()) {
        std::cerr << "Failed to open the file for reading." << std::endl;
    }

    std::string line;
    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        int id;
        double x,y,rx,ry;
        iss>>id>>x>>y>>rx>>ry;
        ids.push_back(id);
        //cout<<rx<<' '<<ry<<' '<<atan2(ry,rx)<<' '<<atan2(ry,rx)/Pi*180<<endl;
        actions.push_back(make_pair(x,y));
        rotates.push_back(atan2(ry,rx));
        //cout<<line<<endl;
    }
}
void read(int ep,vector<int>&ids,vector<pair<double,double>>& actions,vector<double>&rotates){
    
    std::ostringstream oss;
    oss <<"dataset/result_"<<ep<<".txt";
    
    //cout<<oss.str()<<endl;
    std::ifstream inputFile(oss.str());

    if (!inputFile.is_open()) {
        std::cerr << "Failed to open the file for reading." << std::endl;
    }

    std::string line;
    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        int id;
        double x,y,rx,ry;
        iss>>id>>x>>y>>rx>>ry;
        ids.push_back(id);
        //cout<<rx<<' '<<ry<<' '<<atan2(ry,rx)<<' '<<atan2(ry,rx)/Pi*180<<endl;
        actions.push_back(make_pair(x,y));
        rotates.push_back(atan2(ry,rx));
        //cout<<line<<endl;
    }
}
bool cmp1(pair<vector<pair<double,double>>,int> a, pair<vector<pair<double,double>>,int> b){
    double min1x=1e10,min2x=1e10;
    for(auto point :a.first)
        min1x=min(min1x,point.first);
    for(auto point :b.first)
        min2x=min(min2x,point.first);
    return min1x<min2x;
}
pair<double,double> rotatePoint(const pair<double,double> p, double angle) {
    pair<double,double> result;

    result.first = p.first * cos(angle) - p.second * sin(angle);
    result.second = p.first * sin(angle) + p.second * cos(angle) ;

    return result;
}
pair<double,double> translatePoint(const pair<double,double> p, pair<double,double> q) {
    pair<double,double> result;

    result.first = p.first + q.first ;
    result.second = p.second + q.second  ;

    return result;
}

double calculatePolygonArea(const std::vector<pair<double,double>>& vertices) {
    int n = vertices.size();
    double area = 0.0;

    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n; // 下一个顶点的索引

        area += (vertices[i].first * vertices[j].second - vertices[j].first * vertices[i].second);
    }

    area = 0.5 * std::abs(area);
    return area;
}

double getArea(PathsD p){
    double ans=0;
    for(auto path:p){
        std::vector<pair<double,double>> q;
        for(int i=0;i<path.size();i++){
            q.push_back(make_pair(path[i].x,path[i].y));
        }
        ans+=calculatePolygonArea(q);
    }
    //cout<<ans<<endl;
    return ans;
}
void show(PathsD d){
    for(auto p:d){
        cout<<"[";
        for(auto q:p)
            cout<<"["<<q.x<<","<<q.y<<"],";
        cout<<"]\n";
    }
}
PathsD toPathsD(vector<pair<double,double>>a){
    PathsD subject;
    PathD c;
    for(auto p:a){
        c.push_back((PointD){p.first,p.second});
    }
    subject.push_back(c);
    return subject;
}

double areapoly(vector<pair<double,double>>a,vector<pair<double,double>>b){
    
    PathsD subject, clip, solution;
    subject=toPathsD(a);
    clip=toPathsD(b);
    solution = Intersect(subject, clip, FillRule::NonZero);
    return getArea(solution);
}
bool cmp2(pair<vector<pair<double,double>>,int> a, pair<vector<pair<double,double>>,int> b){
    double min1y=1e10,min2y=1e10;
    for(auto point :a.first)
        min1y=min(min1y,point.second);
    for(auto point :b.first)
        min2y=min(min2y,point.second);
    return min1y<min2y;
}
void workx(vector<int> ids,vector<double>rotates,vector<pair<double,double>> & actions,vector<vector<pair<double,double>>> polys){
    Tree tree(2,0,3*ids.size(),false);
    vector<pair<vector<pair<double,double>>,int>>polygons;
    for(int ep=0;ep<ids.size();ep++){
        //cout<<ids[ep]<<' '<<rotates[ep]<<endl;
        vector<pair<double,double>> newpoly;
        for(auto p : polys[ids[ep]])
            newpoly.push_back(translateP(rotateP(p,rotates[ep]),actions[ep]));
        // cout<<"id.append([";
        // for(auto p : newpoly)
        //     cout<<"["<<p.first<<','<<p.second<<"],";
        // cout<<"])\n";
        polygons.push_back(make_pair(newpoly,ep));
    }
    double aabb[ids.size()][4];
    sort(polygons.begin(),polygons.end(),cmp1);
    for(int i=0;i<ids.size();i++){
        pair<vector<pair<double,double>>,int>poly=polygons[i];
        // cout<<poly.second<<endl;
        double mx=1e10,my=1e10,Mx=-1e10,My=-1e10;
        for (auto point:poly.first){
            double x = point.first,y=point.second;
            mx=min(mx,x);
            my=min(my,y);
            Mx=max(Mx,x);
            My=max(My,y);
            aabb[i][0]=mx;
        }
        aabb[i][0]=mx;
        aabb[i][1]=Mx;
        aabb[i][2]=my;
        aabb[i][3]=My;
        // cout<<poly.second<<' '<<mx<<' '<<Mx<<' '<<my<<' '<<My<<endl;

        unsigned int index = i;
        std::vector<double> lowerBound({mx, my});
        std::vector<double> upperBound({Mx, My});
        tree.insertParticle(index, lowerBound, upperBound);
    }
    for(int i=0;i<ids.size();i++){
        // cout<<"begin"<<i<<endl;
        double step=600;
        double cnt=0;
        double mx,Mx,my,My;
        mx=aabb[i][0];
        Mx=aabb[i][1];
        my=aabb[i][2];
        My=aabb[i][3];
        double eps=0.001;
        while(step >= eps){
                
            double minx=mx+cnt;
            double maxx=Mx+cnt;
            if(minx-step>=0){
                bool ins = false;
                vector<pair<double,double>> translated = translate(polygons[i].first,make_pair(cnt-step,0));
                std::vector<double> lowerBound({minx-step, my});
                std::vector<double> upperBound({maxx-step, My});
                aabb::AABB aabb(lowerBound, upperBound);
                std::vector<unsigned int> may = tree.query(aabb);
                if(may.size()){
                    for(auto id : may){
                        if(id == i)
                            continue;
                        if(areapoly(polygons[id].first,translated)>1e-9){
                            ins=true;
                            break;
                        }
                    }
                }
                if(ins == false)
                    cnt-=step;
            }
            step/=2;
        }
        // cout<<"end"<<i<<' '<<cnt<<endl;
        unsigned int index = i;
        tree.removeParticle(index);
        std::vector<double> lowerBound({mx+cnt+eps, my});
        std::vector<double> upperBound({Mx+cnt+eps, My});
        tree.insertParticle(index, lowerBound, upperBound);
        polygons[i].first = translate(polygons[i].first, make_pair(cnt+eps,0));
        actions[polygons[i].second].first+=cnt+eps;
    }
    
    // for(int ep=0;ep<ids.size();ep++){
    //     //cout<<ids[ep]<<' '<<rotates[ep]<<endl;
    //     vector<pair<double,double>> newpoly;
    //     for(auto p : polys[ids[ep]])
    //         newpoly.push_back(translateP(rotateP(p,rotates[ep]),actions[ep]));
    //     cout<<"id.append([";
    //     for(auto p : newpoly)
    //         cout<<"["<<p.first<<','<<p.second<<"],";
    //     cout<<"])\n";
    // }
    return ;
}

void worky(vector<int> ids,vector<double>rotates,vector<pair<double,double>> & actions,vector<vector<pair<double,double>>> polys){
    Tree tree(2,0,ids.size(),false);
    vector<pair<vector<pair<double,double>>,int>>polygons;
    for(int ep=0;ep<ids.size();ep++){
        //cout<<ids[ep]<<' '<<rotates[ep]<<endl;
        vector<pair<double,double>> newpoly;
        for(auto p : polys[ids[ep]])
            newpoly.push_back(translateP(rotateP(p,rotates[ep]),actions[ep]));
        // cout<<"id.append([";
        // for(auto p : newpoly)
        //     cout<<"["<<p.first<<','<<p.second<<"],";
        // cout<<"])\n";
        polygons.push_back(make_pair(newpoly,ep));
    }
    double aabb[ids.size()][4];
    sort(polygons.begin(),polygons.end(),cmp1);
    for(int i=0;i<ids.size();i++){
        pair<vector<pair<double,double>>,int>poly=polygons[i];
        // cout<<poly.second<<endl;
        double mx=1e10,my=1e10,Mx=-1e10,My=-1e10;
        for (auto point:poly.first){
            double x = point.first,y=point.second;
            mx=min(mx,x);
            my=min(my,y);
            Mx=max(Mx,x);
            My=max(My,y);
            aabb[i][0]=mx;
        }
        aabb[i][0]=mx;
        aabb[i][1]=Mx;
        aabb[i][2]=my;
        aabb[i][3]=My;
        // cout<<poly.second<<' '<<mx<<' '<<Mx<<' '<<my<<' '<<My<<endl;

        unsigned int index = i;
        std::vector<double> lowerBound({mx, my});
        std::vector<double> upperBound({Mx, My});
        tree.insertParticle(index, lowerBound, upperBound);
    }
    for(int i=0;i<ids.size();i++){
        // cout<<"begin"<<i<<endl;
        double step=600;
        double cnt=0;
        double mx,Mx,my,My;
        mx=aabb[i][0];
        Mx=aabb[i][1];
        my=aabb[i][2];
        My=aabb[i][3];
        double eps=0.001;
        while(step >= eps){
                
            double miny=my+cnt;
            double maxy=My+cnt;
            if(miny-step>=0){
                bool ins = false;
                vector<pair<double,double>> translated = translate(polygons[i].first,make_pair(0,cnt-step));
                std::vector<double> lowerBound({mx, miny-step});
                std::vector<double> upperBound({Mx, maxy-step});
                aabb::AABB aabb(lowerBound, upperBound);
                std::vector<unsigned int> may = tree.query(aabb);
                if(may.size()){
                    for(auto id : may){
                        if(id == i)
                            continue;
                        if(areapoly(polygons[id].first,translated)>1e-9){
                            ins=true;
                            break;
                        }
                    }
                }
                if(ins == false)
                    cnt-=step;
            }
            step/=2;
        }
        // cout<<"end"<<i<<' '<<cnt<<endl;
        unsigned int index = i;
        tree.removeParticle(index);
        std::vector<double> lowerBound({mx, my+cnt+eps});
        std::vector<double> upperBound({Mx, My+cnt+eps});
        tree.insertParticle(index, lowerBound, upperBound);
        polygons[i].first = translate(polygons[i].first, make_pair(0,cnt+eps));
        actions[polygons[i].second].second+=cnt+eps;
    }
    
    // for(int ep=0;ep<ids.size();ep++){
    //     //cout<<ids[ep]<<' '<<rotates[ep]<<endl;
    //     vector<pair<double,double>> newpoly;
    //     for(auto p : polys[ids[ep]])
    //         newpoly.push_back(translateP(rotateP(p,rotates[ep]),actions[ep]));
    //     cout<<"id.append([";
    //     for(auto p : newpoly)
    //         cout<<"["<<p.first<<','<<p.second<<"],";
    //     cout<<"])\n";
    // }
    return ;
}



double getms(){
    
    auto now = std::chrono::system_clock::now();
    auto millisec = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    auto ms = millisec.time_since_epoch().count();

    // std::cout << "Milliseconds since epoch: " << ms << std::endl;
    return ms;
}

vector<pair<double,double>> get_axes(vector<pair<double,double>> polygon,bool t=true){
    vector<pair<double,double>>axes;
    int n=polygon.size();
    for(int i=0;i<n;i++){
        double x=polygon[(i + 1)%n].first - polygon[i].first;
        double y=polygon[(i + 1)%n].second - polygon[i].second;
        double len=sqrt(x*x+y*y);
        if(len<1e-6)continue;
        x/=len;
        y/=len;
        if(t)
            axes.push_back(make_pair(-y,x));
        else
            axes.push_back(make_pair(y,-x));
    }
    return axes;
}
double project_polygon_onto_vector(PathsD polygon, pair<double,double> vector){

    double ans=-1e10;
    for(auto paths:polygon){
        double min_projection=1e10;
        double max_projection=-1e10;
        // 计算每个顶点在法向量上的投影
        for(auto vertex:paths){
            double projection = vertex.x * vector.first + vertex.y * vector.second;
            min_projection = min(min_projection,projection);
            max_projection = max(max_projection,projection);
        }
        ans=max(ans,max_projection-min_projection);
    }
    return ans;
}
pair<double,double> project_polygon_onto_vector2(PathsD polygon, pair<double,double> vector){

    double ans=-1e10,ans1=0,ans2=0;
    for(auto paths:polygon){
        double min_projection=1e10;
        double max_projection=-1e10;
        // 计算每个顶点在法向量上的投影
        for(auto vertex:paths){
            double projection = vertex.x * vector.first + vertex.y * vector.second;
            min_projection = min(min_projection,projection);
            max_projection = max(max_projection,projection);
        }
        if(max_projection-min_projection>ans){
            ans=max(ans,max_projection-min_projection);
            ans1=max_projection;
            ans2=min_projection;
        }
    }
    return make_pair(ans2,ans1);
}
pair<double,double> project_polygon_onto_vector2(vector<pair<double,double>> polygon, pair<double,double> vector){

    double ans=-1e10,ans1=0,ans2=0;
    
    double min_projection=1e10;
    double max_projection=-1e10;
    // 计算每个顶点在法向量上的投影
    for(auto vertex:polygon){
        double projection = vertex.first * vector.first + vertex.second * vector.second;
        min_projection = min(min_projection,projection);
        max_projection = max(max_projection,projection);
    }
    if(max_projection-min_projection>ans){
        ans=max(ans,max_projection-min_projection);
        ans1=max_projection;
        ans2=min_projection;
    }
    
    return make_pair(ans2,ans1);
}
pair<double,double> separating_axis_theorem(vector<pair<double,double>> poly1,
                                        vector<pair<double,double>> poly2,
                                        PathsD intersectionPoly){
    vector<pair<double,double>> axes = get_axes(poly1);
    vector<pair<double,double>> axes2 = get_axes(poly2, 0);
    axes.insert(axes.end(), axes2.begin(), axes2.end());
    
    if(getArea(intersectionPoly)<1e-9)
        return make_pair(0,0);
    double minproj = 1e10;
    pair<double,double> minaxes = make_pair(0,0);
    for(auto axis : axes){
       double proj = project_polygon_onto_vector(intersectionPoly, axis);
        if (proj<minproj){
            minproj = proj;
            minaxes = axis;
        }
    }
    // 如果所有分离轴都有重叠，说明发生了碰撞
    // print(minproj,minaxes)
    return make_pair(minaxes.first*minproj, minaxes.second*minproj);
}
PathsD insertPolys(vector<pair<double,double>>a,vector<pair<double,double>>b){
    
    PathsD subject, clip, solution;
    subject=toPathsD(a);
    clip=toPathsD(b);
    solution = Intersect(subject, clip, FillRule::NonZero);
    return solution;
}
void rm_overlap(vector<int> ids,vector<double>rotates,
                vector<pair<double,double>> & actions,
                vector<vector<pair<double,double>>> polys,
                double w,double h,double eps=0.01){
    vector<pair<vector<pair<double,double>>,int>>polygons;
    for(int ep=0;ep<ids.size();ep++){
        //cout<<ids[ep]<<' '<<rotates[ep]<<endl;
        vector<pair<double,double>> newpoly;
        for(auto p : polys[ids[ep]])
            newpoly.push_back(translateP(rotateP(p,rotates[ep]),actions[ep]));
        // cout<<"id.append([";
        // for(auto p : newpoly)
        //     cout<<"["<<p.first<<','<<p.second<<"],";
        // cout<<"])\n";
        polygons.push_back(make_pair(newpoly,ep));
    }
    //std::cout << "polygon push done" << std::endl;
    //std::cout << "actions[0] = " << actions[0].first << ", " << actions[0].second << std::endl;

    double aabb[ids.size()][4];
    for(int i=0;i<ids.size();i++){
        pair<vector<pair<double,double>>,int>poly=polygons[i];
        // cout<<poly.second<<endl;
        double mx=1e10,my=1e10,Mx=-1e10,My=-1e10;
        for (auto point:poly.first){
            double x = point.first,y=point.second;
            mx=min(mx,x);
            my=min(my,y);
            Mx=max(Mx,x);
            My=max(My,y);
            aabb[i][0]=mx;
        }
        aabb[i][0]=mx;
        aabb[i][1]=Mx;
        aabb[i][2]=my;
        aabb[i][3]=My;
        // cout<<poly.second<<' '<<mx<<' '<<Mx<<' '<<my<<' '<<My<<endl;
    }

    //std::cout << "aabb init done" << std::endl;
    //std::cout << "actions[0] = " << actions[0].first << ", " << actions[0].second << std::endl;
    //.......................................
    for(int ep=0;ep<2000;ep++){
        //cout<<"ep "<<ep<<endl;
        double v[ids.size()][2]={};
        bool inserted=false;
        Tree tree(2,0,3*ids.size(),false);
        for(int i=0;i<ids.size();i++){
            double mx,Mx,my,My;
            mx=aabb[i][0];
            Mx=aabb[i][1];
            my=aabb[i][2];
            My=aabb[i][3];
            unsigned int index = i;
            std::vector<double> lowerBound({mx, my});
            std::vector<double> upperBound({Mx, My});
            tree.insertParticle(index, lowerBound, upperBound);
        }

        //std::cout << "actions[0] = " << actions[0].first << ", " << actions[0].second << std::endl;
        //std::cout << "v[0][0] = " << v[0][0] << ", v[0][1] = " << v[0][1] << std::endl;
        for(int i=0;i<ids.size();i++){
            double mx,Mx,my,My;
            mx=aabb[i][0];
            Mx=aabb[i][1];
            my=aabb[i][2];
            My=aabb[i][3];
            std::vector<double> lowerBound({mx, my});
            std::vector<double> upperBound({Mx, My});
            aabb::AABB aabb(lowerBound, upperBound);
            std::vector<unsigned int> may = tree.query(aabb);
            for(int id:may){
                if(id<=i)continue;
                if(areapoly(polygons[id].first,polygons[i].first)>eps){
                    inserted=true;
                    pair<double,double> d=separating_axis_theorem(polygons[i].first,polygons[id].first,insertPolys(polygons[i].first,polygons[id].first));
                    double dx=d.first,dy=d.second;
                    pair<double,double> pii=project_polygon_onto_vector2(polygons[i].first,d);
                    double pi=pii.first,Pi=pii.second;
                    pair<double,double> piid=project_polygon_onto_vector2(polygons[id].first,d);
                    double pid=piid.first,Pid=piid.second;
                    pair<double,double> pp=project_polygon_onto_vector2(insertPolys(polygons[i].first,polygons[id].first),d);
                    double p=pp.first,P=pp.second;
                    double xa=0,ya=0,xb=0,yb=0;
                    for(auto point:polygons[i].first){
                        xa+=point.first;
                        ya+=point.second;
                    }
                    xa/=polygons[i].first.size();
                    ya/=polygons[i].first.size();
                    for(auto point:polygons[id].first){
                        xb+=point.first;
                        yb+=point.second;
                    }
                    xb/=polygons[id].first.size();
                    yb/=polygons[id].first.size();
                    double xab=xb-xa;
                    double yab=yb-ya;
                    double area1=calculatePolygonArea(polygons[i].first);
                    double area2=calculatePolygonArea(polygons[id].first);
                    //std::cout << id << " " << i << std::endl;
                    //std::cout << dx << " " << dy << std::endl;
                    if(true){
                        v[i][0]+=area2*dx/(area1+area2);
                        v[i][1]+=area2*dy/(area1+area2);
                        v[id][0]-=area1*dx/(area1+area2);
                        v[id][1]-=area1*dy/(area1+area2);
                    }
                    else{
                        v[i][0]-=area2*dx/(area1+area2);
                        v[i][1]-=area2*dy/(area1+area2);
                        v[id][0]+=area1*dx/(area1+area2);
                        v[id][1]+=area1*dy/(area1+area2);
                    }
                }
            }
        }

        //std::cout << "inserted = " << inserted << std::endl;
        //std::cout << "actions[0] = " << actions[0].first << ", " << actions[0].second << std::endl;
        //std::cout << "v[0][0] = " << v[0][0] << ", v[0][1] = " << v[0][1] << std::endl;
        bool inbox=true;
        for(int i=0;i<ids.size();i++){
            double mx,Mx,my,My;
            mx=aabb[i][0];
            Mx=aabb[i][1];
            my=aabb[i][2];
            My=aabb[i][3];
            if(mx<0){
                v[i][0]-=mx;
                inbox=false;
            }
            if(Mx>w){
                v[i][0]-=Mx-w;
                inbox=false;
            }
            if(my<0){
                v[i][1]-=my;
                inbox=false;
            }
            if(My>h){
                v[i][1]-=My-h;
                inbox=false;
            }
        }

        //std::cout << "inbox = " << inbox << std::endl;
        //std::cout << "actions[0] = " << actions[0].first << ", " << actions[0].second << std::endl;
        //std::cout << "v[0][0] = " << v[0][0] << ", v[0][1] = " << v[0][1] << std::endl;
        for(int i=0;i<ids.size();i++){
            double mx,Mx,my,My;
            mx=aabb[i][0];
            Mx=aabb[i][1];
            my=aabb[i][2];
            My=aabb[i][3];
            mx+=v[i][0];
            Mx+=v[i][0];
            my+=v[i][1];
            My+=v[i][1];
            actions[i].first+=v[i][0];
            actions[i].second+=v[i][1];
            aabb[i][0]=mx;
            aabb[i][1]=Mx;
            aabb[i][2]=my;
            aabb[i][3]=My;
            polygons[i].first = translate(polygons[i].first,make_pair(v[i][0],v[i][1]));
        }

        //std::cout << "inserted = " << inserted << std::endl;
        //std::cout << "actions[0] = " << actions[0].first << ", " << actions[0].second << std::endl;
        //std::cout << "v[0][0] = " << v[0][0] << ", v[0][1] = " << v[0][1] << std::endl;
        if(inserted==false){
            double mx=1e10,my=1e10;
            for(int i=0;i<ids.size();i++){
                for(auto point:polygons[i].first){
                    mx=min(mx,point.first);
                    my=min(my,point.second);
                }
            }
            for(int i=0;i<ids.size();i++){
                actions[i].first-=mx;
                actions[i].second-=my;
            }
            return ;
        }
    }
}

/*
vector<pair<double,double>> rm_spacing(vector<int> ids,vector<double>rotates,vector<pair<double,double>>actions,vector<vector<pair<double,double>>> polys
                            ,double width,double height,double OverlapEps){//,double Range,double Overlapeps){

    rm_overlap(ids,rotates,actions,polys,width,height,OverlapEps);
    //workx(ids,rotates,actions,polys);
    //worky(ids,rotates,actions,polys);
    return actions;
}
*/


vector<pair<double,double>> rm_spacing(vector<int> ids,vector<double>rotates,vector<pair<double,double>>actions,vector<vector<pair<double,double>>> polys, double width, double height, double OverlapEps){//,double Range,double Overlapeps){
    workx(ids,rotates,actions,polys);
    worky(ids,rotates,actions,polys);
    return actions;
}

vector<vector<pair<double,double>>> rm_spacing_all(vector<vector<int>> ids, vector<vector<double>> rotates, vector<vector<pair<double,double>>> actions, vector<vector<pair<double,double>>> polys, double width, double height, double OverlapEps){
    // 
    // std::cout << "ids.size() = " << ids.size() << std::endl;
    // std::cout << "rotates.size() = " << rotates.size() << std::endl;
    // std::cout << "actions.size() = " << actions.size() << std::endl;
    // std::cout << "polys.size() = " << polys.size() << std::endl;
    // std::cout << "width = " << width << std::endl;
    // std::cout << "height = " << height << std::endl;
    // std::cout << "OverlapEps = " << OverlapEps << std::endl;

    #pragma omp parallel for
    for (int i = 0; i < ids.size(); ++i) {
        // rm_overlap(ids[i], rotates[i], actions[i], polys, width, height, OverlapEps);
        
        workx(ids[i], rotates[i], actions[i], polys);
        worky(ids[i], rotates[i], actions[i], polys);
        workx(ids[i], rotates[i], actions[i], polys);
        worky(ids[i], rotates[i], actions[i], polys);
        workx(ids[i], rotates[i], actions[i], polys);
    }
    return actions;
}


PYBIND11_MODULE(rmspacing, m) {
    m.doc() = "pybind11 rm_spacing"; // optional module docstring
    m.def("rm_spacing_all", &rm_spacing_all, "A function that removes spacing");
}



/*
int main() {
    vector<vector<pair<double,double>>> polygon=open(440);
    int rangel=0,ranger=32;
    double s0=time(0);
    for(int i=rangel;i<ranger;i++){
        double s = getms();
        vector<int>ids;
        vector<pair<double,double>> actions;
        vector<double>rotates;
        read2(i,ids,actions,rotates);
        //cout<<i<<' '<<ids.size()<<' '<<actions.size()<<' '<<rotates.size()<<endl;
        vector<pair<double,double>> actions1=rm_spacing(ids,rotates,actions,polygon);
        //std::ostringstream oss;
        //oss << "ans/"<<i<<".txt";
        //std::ofstream outputFile(oss.str());
            
        double t = getms();
        cout<<"time "<<t-s<<endl;
    }
    cout<<"time0 "<<time(0)-s0<<endl;
    return 0;
}
*/