function Label=SLIC(img,s,errTh,wDs)
% 基于KMeans的超像素分割
% img为输入图像，维度不限，最大值为255
% s x s为超像素尺寸
% errTh为控制迭代结束的联合向量残差上限
m=size(img,1);
n=size(img,2);
 
%% 计算栅格顶点与中心的坐标
h=floor(m/s);%栅格在长上个数
w=floor(n/s);%栅格在宽上个数
rowR=floor((m-h*s)/2); %行多余部分首尾均分
colR=floor((n-w*s)/2);%行多余部分首尾均分
rowStart=(rowR+1):s:(m-s+1);%每个超像素行起点
rowStart(1)=1;
rowEnd=rowStart+s;%每个超像素列终点
rowEnd(1)=rowR+s;
rowEnd(end)=m;
colStart=(colR+1):s:(n-s+1);%每个超像素列起点
colStart(1)=1;
colEnd=colStart+s;%每个超像素列终点
colEnd(1)=colR+s;
colEnd(end)=n;
rowC=floor((rowStart+rowEnd-1)/2);%超像素的行中心
colC=floor((colStart+colEnd-1)/2);%超像素的列中心
% 显示划分结果
temp=zeros(m,n);
temp(rowStart,:)=1;%划栅格列线
temp(:,colStart)=1;%划栅格行线
for i=1:h
    for j=1:w
        temp(rowC(i),colC(j))=1;%对中心点进行标记
    end
end
figure,imshow(temp);
imwrite(temp,'栅格.bmp');
 
%% 计算梯度图像，使用sobel算子和欧式距离
img=double(img)/255;%图像归一化
%取rgb每个通道分量
r=img(:,:,1);
g=img(:,:,2);
b=img(:,:,3);
Y=0.299 * r + 0.587 * g + 0.114 * b;%图像融合为灰度图
%sobel梯度算子
f1=fspecial('sobel');
f2=f1';
%x、y方向上滤波
gx=imfilter(Y,f1);
gy=imfilter(Y,f2);
%梯度图
G=sqrt(gx.^2+gy.^2); 

%% 选择栅格中心点3*3邻域中梯度最小点作为起始点
rowC_std=repmat(rowC',[1,w]);%行中心点复制w次
colC_std=repmat(colC,[h,1]);%列中心点复制h次
rowC=rowC_std;
colC=colC_std;
for i=1:h
    for j=1:w
        %取栅格中心点的3x3邻域
        block=G(rowC(i,j)-1:rowC(i,j)+1,colC(i,j)-1:colC(i,j)+1);
        %取梯度最小的值与坐标
        [minVal,idxArr]=min(block(:));
        jOffset=floor((idxArr(1)+2)/3);
        iOffset=idxArr(1)-3*(jOffset-1);
        %更新栅格中心点为梯度最小点
        rowC(i,j)=rowC(i,j)+iOffset;
        colC(i,j)=colC(i,j)+jOffset;
    end
end
 
%% KMeans超像素分割
Label=zeros(m,n)-1;%label起始设为-1
dis=Inf*ones(m,n);%距离设置为无限大
M=reshape(img,m*n,size(img,3)); %像素值重排为3列(按列优先)
% 联合色域值和空域值
colorC=zeros(h,w,size(img,3));
for i=1:h
    for j=1:w
        %将栅格中心点对应img灰度值放到ColorC中
        colorC(i,j,:)=img(rowC(i),colC(j),:);
    end
end
%前3维为img中心点灰度值，后两维为对应坐标
uniMat=cat(3,colorC,rowC,colC);
%uniMat为前三列为质心，后两列为坐标
uniMat=reshape(uniMat,h*w,size(img,3)+2);
iter=1;%迭代次数
while(1)
     uniMat_old=uniMat;
%     rowC_old=rowC;
%     colC_old=colC;
    for k=1:h*w
        c=floor((k-1)/h)+1;%先行后列的遍历
        r=k-h*(c-1);
        rowCidx=rowC(r,c);%记录每一个中心点坐标
        colCidx=colC(r,c);
        %根据栅格取每个超像素的四个顶点
        rowStart=max(1,rowC_std(r,c)-s);
        rowEnd=min(m,rowC_std(r,c)+s-1);
        colStart=max(1,colC_std(r,c)-s);
        colEnd=min(n,colC_std(r,c)+s);
        %colorC=uniMat(k,1:size(img,3));
        colorC=M((colCidx-1)*m+rowCidx,:);%中心点对应图像灰度值赋给colorC
        %搜索区域限定在超像素格内
        for i=rowStart:rowEnd
            for j=colStart:colEnd
                colorCur=M((j-1)*m+i,:);%存放对应点的像素灰度值
                dc=norm(colorC-colorCur);%计算灰度值差
                ds=norm([i-rowCidx,j-colCidx]);%计算坐标差
                d=dc^2+wDs*(ds/s)^2;%统一坐标距离后再计算距离和
                %更新距离与标签
                if d<dis(i,j)
                    dis(i,j)=d;
                    Label(i,j)=k;
                end
            end
        end
    end
    
    %显示聚类结果
    temp=mod(Label,20)+1;
    figure;
    %图像，颜色映射表，白色，颜色映射方式（伪随机或按序）
    imagesc(label2rgb(temp-1,'jet','w','shuffle')) ;
    axis image ; %使图像紧凑以及x，y坐标比例一致
    axis off ;
    
    % 录制gif
    F=getframe(gcf);%捕获坐标区或figure作为影片帧
    I=frame2im(F);%返回与影片帧关联的图像数据
    [I,map]=rgb2ind(I,256);%将 RGB 图像转换为索引图像
    if iter == 1
        imwrite(I,map,'test.gif','gif','Loopcount',inf,'DelayTime',0.2);
    else
        imwrite(I,map,'test.gif','gif','WriteMode','append','DelayTime',0.2);
    end
    iter=iter+1;
    
    % 更新聚类中心
    colorC=zeros(h,w,size(img,3));
    for k=1:h*w
        num=0;%记录超像素格内有多少元素
        sumColor=zeros(1,size(img,3));  %记录灰度和值  
        sumR=0;%记录行坐标和值
        sumC=0;%记录列坐标和值
        c=floor((k-1)/h)+1;%c，r为第k块超像素大致坐标
        r=k-h*(c-1);
        rowCidx=rowC_std(r,c);%超像素块中心点坐标
        colCidx=colC_std(r,c);
        rowStart=max(1,rowCidx-s);
        rowEnd=min(m,rowCidx+s-1);
        colStart=max(1,colCidx-s);
        colEnd=min(n,colCidx+s);
        %统计类中各项指标和值
        for row=rowStart:rowEnd
            for col=colStart:colEnd
                if Label(row,col)==k
                    num=num+1;
                    sumR=sumR+row;
                    sumC=sumC+col;
                    color=reshape(img(row,col,:),1,size(img,3));
                    sumColor=sumColor+color;
                end
            end
        end
        colorC(r,c,:)=sumColor/num;%更新后灰度均值
        rowC(r,c)=round(sumR/num);%更新后行均值
        colC(r,c)=round(sumC/num);%更新后列均值
    end
    %计算迭代后质心差，与Kmeans一致
    uniMat=cat(3,colorC,rowC,colC);
    %uniMat为前三列为质心，后两列为坐标
    uniMat=reshape(uniMat,h*w,size(img,3)+2);
    diff=uniMat-uniMat_old;
    diff(:,1:2)=sqrt(wDs)*diff(:,1:2)/s;
    err=norm(diff)/sqrt(h*w);
    if err<errTh %残差低于阈值，结束迭代
        break;
    end
end
 
%% 后处理， 按照边界接触点数最多原则分配小连通域的标签
for k=1:h*w
    c=floor((k-1)/h)+1;
    r=k-h*(c-1);
    rowCidx=rowC_std(r,c);%第k个像素块的中心点
    colCidx=colC_std(r,c);
    rowStart=max(1,rowCidx-s);%第k个像素块的四周顶点
    rowEnd=min(m,rowCidx+s-1);
    colStart=max(1,colCidx-s);
    colEnd=min(n,colCidx+s);
    block=Label(rowStart:rowEnd,colStart:colEnd);%取出四端点范围内的标签
    block(block~=k)=0;%若标签不为k，则设置为0
    block(block==k)=1;%若标签为k，则设置为1
    %标签为0的像素构成背景，为1的像素构成一个对象，为 2 的像素构成另一个对象
    label=bwlabel(block);%对二维二值图像中的连通分量进行标注
    szlabel=max(label(:)); %记录存在几个连通域
    bh=rowEnd-rowStart+1;%block的长
    bw=colEnd-colStart+1;  %block的宽
    
    if szlabel<2  %无伴生连通域，略过
        continue;
    end
    
    labelC=label(rowCidx-rowStart+1,colCidx-colStart+1); %主连通域的标记值
    top=max(1,rowStart-1);%记录超像素的四个端点
    bottom=min(m,rowEnd+1);
    left=max(1,colStart-1);
    right=min(n,colEnd+1);
    for i=1:szlabel %遍历连通域
        if i==labelC %主连通域不处理
            continue;
        end
        %生成一个外扩一圈的marker，标记哪些点已经被统计过接触情况
        marker=zeros(bottom-top+1,right-left+1); 
        bw=label;
        bw(bw~=i)=0;
        bw(bw==i)=1; %当前连通域标记图
        contourBW=bwperim(bw); %求取外轮廓
        %figure,imshow(contourBW);
        idxArr=find(double(contourBW)==1);%返回为1像素的列坐标
        labelArr=zeros(4*length(idxArr),1);  %记录轮廓点的4邻域点标记值的向量
        num=0;
        for idx=1:size(idxArr) %遍历轮廓点,统计其4邻域点的标记值
            bc=floor((idxArr(idx)-1)/bh)+1;
            br=idxArr(idx)-bh*(bc-1); %轮廓点在block中的行列信息
            row=br+rowStart-1;
            col=bc+colStart-1; %轮廓点在大图中的行列信息
            rc=[row-1,col;...
                row+1,col;...
                row,col-1;...
                row,col+1];
            for p=1:4
                row=rc(p,1);%第一列
                col=rc(p,2);%第二列
                %如果为另一对象，则不进行统计
                if ~(row>=1 && row<=m && col>=1 && col<=n && Label(row,col)~=k)
                    continue;
                end
                 %若该点未被统计过，则统计
                if marker(row-top+1,col-left+1)==0
                    marker(row-top+1,col-left+1)=1;
                    num=num+1;
                    labelArr(num)=Label(row,col);
                end
            end
        end
        
        labelArr(find(labelArr==0))=[]; %去除零元素
        uniqueLabel=unique(labelArr);%去除标签中拥有的值
        numArr=zeros(length(uniqueLabel),1);
        %计算每个标签个数
        for p=1:length(uniqueLabel)
            idx=find(labelArr==uniqueLabel(p));
            numArr(p)=length(idx);
        end
        idx=find(numArr==max(numArr));
        maxnumLabel=uniqueLabel(idx(1)); %接触最多的标签
        %将接触最多标签赋给未联通像素
        for row=rowStart:rowEnd
            for col=colStart:colEnd
                if bw(row-rowStart+1,col-colStart+1)==0
                    continue;
                end
                Label(row,col)=maxnumLabel;
            end
        end
    end
end
 
% 显示连通域处理后聚类结果
temp=mod(Label,20)+1;
figure;
imagesc(label2rgb(temp-1,'jet','w','shuffle')) ;
axis image ; axis off ;


