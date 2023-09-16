function Label=SLIC(img,s,errTh,wDs)
% ����KMeans�ĳ����طָ�
% imgΪ����ͼ��ά�Ȳ��ޣ����ֵΪ255
% s x sΪ�����سߴ�
% errThΪ���Ƶ������������������в�����
m=size(img,1);
n=size(img,2);
 
%% ����դ�񶥵������ĵ�����
h=floor(m/s);%դ���ڳ��ϸ���
w=floor(n/s);%դ���ڿ��ϸ���
rowR=floor((m-h*s)/2); %�ж��ಿ����β����
colR=floor((n-w*s)/2);%�ж��ಿ����β����
rowStart=(rowR+1):s:(m-s+1);%ÿ�������������
rowStart(1)=1;
rowEnd=rowStart+s;%ÿ�����������յ�
rowEnd(1)=rowR+s;
rowEnd(end)=m;
colStart=(colR+1):s:(n-s+1);%ÿ�������������
colStart(1)=1;
colEnd=colStart+s;%ÿ�����������յ�
colEnd(1)=colR+s;
colEnd(end)=n;
rowC=floor((rowStart+rowEnd-1)/2);%�����ص�������
colC=floor((colStart+colEnd-1)/2);%�����ص�������
% ��ʾ���ֽ��
temp=zeros(m,n);
temp(rowStart,:)=1;%��դ������
temp(:,colStart)=1;%��դ������
for i=1:h
    for j=1:w
        temp(rowC(i),colC(j))=1;%�����ĵ���б��
    end
end
figure,imshow(temp);
imwrite(temp,'դ��.bmp');
 
%% �����ݶ�ͼ��ʹ��sobel���Ӻ�ŷʽ����
img=double(img)/255;%ͼ���һ��
%ȡrgbÿ��ͨ������
r=img(:,:,1);
g=img(:,:,2);
b=img(:,:,3);
Y=0.299 * r + 0.587 * g + 0.114 * b;%ͼ���ں�Ϊ�Ҷ�ͼ
%sobel�ݶ�����
f1=fspecial('sobel');
f2=f1';
%x��y�������˲�
gx=imfilter(Y,f1);
gy=imfilter(Y,f2);
%�ݶ�ͼ
G=sqrt(gx.^2+gy.^2); 

%% ѡ��դ�����ĵ�3*3�������ݶ���С����Ϊ��ʼ��
rowC_std=repmat(rowC',[1,w]);%�����ĵ㸴��w��
colC_std=repmat(colC,[h,1]);%�����ĵ㸴��h��
rowC=rowC_std;
colC=colC_std;
for i=1:h
    for j=1:w
        %ȡդ�����ĵ��3x3����
        block=G(rowC(i,j)-1:rowC(i,j)+1,colC(i,j)-1:colC(i,j)+1);
        %ȡ�ݶ���С��ֵ������
        [minVal,idxArr]=min(block(:));
        jOffset=floor((idxArr(1)+2)/3);
        iOffset=idxArr(1)-3*(jOffset-1);
        %����դ�����ĵ�Ϊ�ݶ���С��
        rowC(i,j)=rowC(i,j)+iOffset;
        colC(i,j)=colC(i,j)+jOffset;
    end
end
 
%% KMeans�����طָ�
Label=zeros(m,n)-1;%label��ʼ��Ϊ-1
dis=Inf*ones(m,n);%��������Ϊ���޴�
M=reshape(img,m*n,size(img,3)); %����ֵ����Ϊ3��(��������)
% ����ɫ��ֵ�Ϳ���ֵ
colorC=zeros(h,w,size(img,3));
for i=1:h
    for j=1:w
        %��դ�����ĵ��Ӧimg�Ҷ�ֵ�ŵ�ColorC��
        colorC(i,j,:)=img(rowC(i),colC(j),:);
    end
end
%ǰ3άΪimg���ĵ�Ҷ�ֵ������άΪ��Ӧ����
uniMat=cat(3,colorC,rowC,colC);
%uniMatΪǰ����Ϊ���ģ�������Ϊ����
uniMat=reshape(uniMat,h*w,size(img,3)+2);
iter=1;%��������
while(1)
     uniMat_old=uniMat;
%     rowC_old=rowC;
%     colC_old=colC;
    for k=1:h*w
        c=floor((k-1)/h)+1;%���к��еı���
        r=k-h*(c-1);
        rowCidx=rowC(r,c);%��¼ÿһ�����ĵ�����
        colCidx=colC(r,c);
        %����դ��ȡÿ�������ص��ĸ�����
        rowStart=max(1,rowC_std(r,c)-s);
        rowEnd=min(m,rowC_std(r,c)+s-1);
        colStart=max(1,colC_std(r,c)-s);
        colEnd=min(n,colC_std(r,c)+s);
        %colorC=uniMat(k,1:size(img,3));
        colorC=M((colCidx-1)*m+rowCidx,:);%���ĵ��Ӧͼ��Ҷ�ֵ����colorC
        %���������޶��ڳ����ظ���
        for i=rowStart:rowEnd
            for j=colStart:colEnd
                colorCur=M((j-1)*m+i,:);%��Ŷ�Ӧ������ػҶ�ֵ
                dc=norm(colorC-colorCur);%����Ҷ�ֵ��
                ds=norm([i-rowCidx,j-colCidx]);%���������
                d=dc^2+wDs*(ds/s)^2;%ͳһ���������ټ�������
                %���¾������ǩ
                if d<dis(i,j)
                    dis(i,j)=d;
                    Label(i,j)=k;
                end
            end
        end
    end
    
    %��ʾ������
    temp=mod(Label,20)+1;
    figure;
    %ͼ����ɫӳ�����ɫ����ɫӳ�䷽ʽ��α�������
    imagesc(label2rgb(temp-1,'jet','w','shuffle')) ;
    axis image ; %ʹͼ������Լ�x��y�������һ��
    axis off ;
    
    % ¼��gif
    F=getframe(gcf);%������������figure��ΪӰƬ֡
    I=frame2im(F);%������ӰƬ֡������ͼ������
    [I,map]=rgb2ind(I,256);%�� RGB ͼ��ת��Ϊ����ͼ��
    if iter == 1
        imwrite(I,map,'test.gif','gif','Loopcount',inf,'DelayTime',0.2);
    else
        imwrite(I,map,'test.gif','gif','WriteMode','append','DelayTime',0.2);
    end
    iter=iter+1;
    
    % ���¾�������
    colorC=zeros(h,w,size(img,3));
    for k=1:h*w
        num=0;%��¼�����ظ����ж���Ԫ��
        sumColor=zeros(1,size(img,3));  %��¼�ҶȺ�ֵ  
        sumR=0;%��¼�������ֵ
        sumC=0;%��¼�������ֵ
        c=floor((k-1)/h)+1;%c��rΪ��k�鳬���ش�������
        r=k-h*(c-1);
        rowCidx=rowC_std(r,c);%�����ؿ����ĵ�����
        colCidx=colC_std(r,c);
        rowStart=max(1,rowCidx-s);
        rowEnd=min(m,rowCidx+s-1);
        colStart=max(1,colCidx-s);
        colEnd=min(n,colCidx+s);
        %ͳ�����и���ָ���ֵ
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
        colorC(r,c,:)=sumColor/num;%���º�ҶȾ�ֵ
        rowC(r,c)=round(sumR/num);%���º��о�ֵ
        colC(r,c)=round(sumC/num);%���º��о�ֵ
    end
    %������������Ĳ��Kmeansһ��
    uniMat=cat(3,colorC,rowC,colC);
    %uniMatΪǰ����Ϊ���ģ�������Ϊ����
    uniMat=reshape(uniMat,h*w,size(img,3)+2);
    diff=uniMat-uniMat_old;
    diff(:,1:2)=sqrt(wDs)*diff(:,1:2)/s;
    err=norm(diff)/sqrt(h*w);
    if err<errTh %�в������ֵ����������
        break;
    end
end
 
%% ���� ���ձ߽�Ӵ��������ԭ�����С��ͨ��ı�ǩ
for k=1:h*w
    c=floor((k-1)/h)+1;
    r=k-h*(c-1);
    rowCidx=rowC_std(r,c);%��k�����ؿ�����ĵ�
    colCidx=colC_std(r,c);
    rowStart=max(1,rowCidx-s);%��k�����ؿ�����ܶ���
    rowEnd=min(m,rowCidx+s-1);
    colStart=max(1,colCidx-s);
    colEnd=min(n,colCidx+s);
    block=Label(rowStart:rowEnd,colStart:colEnd);%ȡ���Ķ˵㷶Χ�ڵı�ǩ
    block(block~=k)=0;%����ǩ��Ϊk��������Ϊ0
    block(block==k)=1;%����ǩΪk��������Ϊ1
    %��ǩΪ0�����ع��ɱ�����Ϊ1�����ع���һ������Ϊ 2 �����ع�����һ������
    label=bwlabel(block);%�Զ�ά��ֵͼ���е���ͨ�������б�ע
    szlabel=max(label(:)); %��¼���ڼ�����ͨ��
    bh=rowEnd-rowStart+1;%block�ĳ�
    bw=colEnd-colStart+1;  %block�Ŀ�
    
    if szlabel<2  %�ް�����ͨ���Թ�
        continue;
    end
    
    labelC=label(rowCidx-rowStart+1,colCidx-colStart+1); %����ͨ��ı��ֵ
    top=max(1,rowStart-1);%��¼�����ص��ĸ��˵�
    bottom=min(m,rowEnd+1);
    left=max(1,colStart-1);
    right=min(n,colEnd+1);
    for i=1:szlabel %������ͨ��
        if i==labelC %����ͨ�򲻴���
            continue;
        end
        %����һ������һȦ��marker�������Щ���Ѿ���ͳ�ƹ��Ӵ����
        marker=zeros(bottom-top+1,right-left+1); 
        bw=label;
        bw(bw~=i)=0;
        bw(bw==i)=1; %��ǰ��ͨ����ͼ
        contourBW=bwperim(bw); %��ȡ������
        %figure,imshow(contourBW);
        idxArr=find(double(contourBW)==1);%����Ϊ1���ص�������
        labelArr=zeros(4*length(idxArr),1);  %��¼�������4�������ֵ������
        num=0;
        for idx=1:size(idxArr) %����������,ͳ����4�����ı��ֵ
            bc=floor((idxArr(idx)-1)/bh)+1;
            br=idxArr(idx)-bh*(bc-1); %��������block�е�������Ϣ
            row=br+rowStart-1;
            col=bc+colStart-1; %�������ڴ�ͼ�е�������Ϣ
            rc=[row-1,col;...
                row+1,col;...
                row,col-1;...
                row,col+1];
            for p=1:4
                row=rc(p,1);%��һ��
                col=rc(p,2);%�ڶ���
                %���Ϊ��һ�����򲻽���ͳ��
                if ~(row>=1 && row<=m && col>=1 && col<=n && Label(row,col)~=k)
                    continue;
                end
                 %���õ�δ��ͳ�ƹ�����ͳ��
                if marker(row-top+1,col-left+1)==0
                    marker(row-top+1,col-left+1)=1;
                    num=num+1;
                    labelArr(num)=Label(row,col);
                end
            end
        end
        
        labelArr(find(labelArr==0))=[]; %ȥ����Ԫ��
        uniqueLabel=unique(labelArr);%ȥ����ǩ��ӵ�е�ֵ
        numArr=zeros(length(uniqueLabel),1);
        %����ÿ����ǩ����
        for p=1:length(uniqueLabel)
            idx=find(labelArr==uniqueLabel(p));
            numArr(p)=length(idx);
        end
        idx=find(numArr==max(numArr));
        maxnumLabel=uniqueLabel(idx(1)); %�Ӵ����ı�ǩ
        %���Ӵ�����ǩ����δ��ͨ����
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
 
% ��ʾ��ͨ����������
temp=mod(Label,20)+1;
figure;
imagesc(label2rgb(temp-1,'jet','w','shuffle')) ;
axis image ; axis off ;


