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
double getms(){
    
    auto now = std::chrono::system_clock::now();
    auto millisec = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    auto ms = millisec.time_since_epoch().count();

    // std::cout << "Milliseconds since epoch: " << ms << std::endl;
    return ms;
}
vector<double> util(vector<int> ids,vector<double>rotates,
                vector<pair<double,double>> & actions,
                vector<vector<pair<double,double>>> polys,
                double eps){
    vector<pair<vector<pair<double,double>>,int>>polygons;
    for(int ep=0;ep<ids.size();ep++){
        vector<pair<double,double>> newpoly;
        for(auto p : polys[ids[ep]])
            newpoly.push_back(translateP(rotateP(p,rotates[ep]),actions[ep]));
        polygons.push_back(make_pair(newpoly,ep));
    }
    double aabb[ids.size()][4];
    for(int i=0;i<ids.size();i++){
        pair<vector<pair<double,double>>,int>poly=polygons[i];
        double mx=1e10,my=1e10,Mx=-1e10,My=-1e10;
        for (auto point:poly.first){
            double x = point.first,y=point.second;
            mx=min(mx,x);
            my=min(my,y);
            Mx=max(Mx,x);
            My=max(My,y);
        }
        aabb[i][0]=mx;
        aabb[i][1]=Mx;
        aabb[i][2]=my;
        aabb[i][3]=My;
    }
    double sumarea=0,maxinsertarea=0,suminsertarea=0,minx=1e10,miny=1e10,maxx=-1e10,maxy=-1e10;

    //.......................................
    
    bool inserted=false;
    Tree tree(2,0,3*ids.size(),false);
    for(int i=0;i<ids.size();i++){
        double mx,Mx,my,My;
        mx=aabb[i][0];
        Mx=aabb[i][1];
        my=aabb[i][2];
        My=aabb[i][3];
        minx=min(mx,minx);
        miny=min(my,miny);
        maxx=max(Mx,maxx);
        maxy=max(My,maxy);
        unsigned int index = i;
        std::vector<double> lowerBound({mx, my});
        std::vector<double> upperBound({Mx, My});
        tree.insertParticle(index, lowerBound, upperBound);
    }
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
        sumarea+=calculatePolygonArea(polygons[i].first);
        for(int id:may){
            if(id<=i)continue;
            double ins=areapoly(polygons[id].first,polygons[i].first);
            if(ins>eps){
                inserted=true;
                suminsertarea+=ins;
                maxinsertarea=max(maxinsertarea,ins);
            }
        }
    }
    // cout<<sumarea<<' '<<maxinsertarea<<' '<<suminsertarea<<' '<<minx<<' '<<miny<<' '<<maxx<<' '<<maxy<<endl;
    vector<double>re;
    re.push_back(sumarea);
    re.push_back(maxinsertarea);
    re.push_back(suminsertarea);
    re.push_back(minx);
    re.push_back(miny);
    re.push_back(maxx);
    re.push_back(maxy);
    return re;
}

vector<vector<double>> cal_util_all(
    vector<vector<int>> ids, 
    vector<vector<double>> rotates, 
    vector<vector<pair<double,double>>> actions, 
    vector<vector<pair<double,double>>> polys, 
    double OverlapEps){
    // 
    // std::cout << "ids.size() = " << ids.size() << std::endl;
    // std::cout << "rotates.size() = " << rotates.size() << std::endl;
    // std::cout << "actions.size() = " << actions.size() << std::endl;
    // std::cout << "polys.size() = " << polys.size() << std::endl;
    // std::cout << "width = " << width << std::endl;
    // std::cout << "height = " << height << std::endl;
    // std::cout << "OverlapEps = " << OverlapEps << std::endl;
    // res size: ids.size() * 7
    std::vector<std::vector<double>> res(ids.size(), std::vector<double>(7, 0));

    #pragma omp parallel for
    for (int i = 0; i < ids.size(); ++i) {
        res[i] = util(ids[i], rotates[i], actions[i], polys, OverlapEps);
    }

    return res;
}


PYBIND11_MODULE(calutil, m) {
    m.doc() = "pybind11 calutil"; // optional module docstring
    m.def("cal_util_all", &cal_util_all, "A function that calculates the util of all polygons.");
}


/*
int main() {
    vector<vector<pair<double,double>>> polygon=open(440);
    int rangel=0,ranger=3;
    double s0=time(0);
    for(int i=rangel;i<ranger;i++){
        double s = getms();
        vector<int>ids;
        vector<pair<double,double>> actions;
        vector<double>rotates;
        read(i,ids,actions,rotates);
        //cout<<i<<' '<<ids.size()<<' '<<actions.size()<<' '<<rotates.size()<<endl;
        //vector<pair<double,double>> actions1=rm_spacing(ids,rotates,actions,polygon,4000,1200,0.5);
        util(ids,rotates,actions,polygon,0.5);
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
