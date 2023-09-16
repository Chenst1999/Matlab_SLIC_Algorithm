close all;
clear all;
clc;
I=imread(uigetfile('.jpg'));
%figure,imshow(I);
 
s=80;
errTh=0.1;
wDs=0.5^2;
Label=SLIC(I,s,errTh,wDs);
 
%% ��ʾ����
marker=zeros(size(Label));
[m,n]=size(Label);
for i=1:m
    for j=1:n
        top=Label(max(1,i-1),j);
        bottom=Label(min(m,i+1),j);
        left=Label(i,max(1,j-1));
        right=Label(i,min(n,j+1));
        if ~(top==bottom && bottom==left && left==right)
            marker(i,j)=1;
        end
    end
end
figure,imshow(marker);
 
I2=I;
for i=1:m
    for j=1:n
        if marker(i,j)==1
            I2(i,j,:)=0;
        end
    end
end

figure,
subplot(1,2,1);imshow(I);title('Lenaԭͼ','fontsize',30);
subplot(1,2,2);imshow(I2);title('SLIC�����','fontsize',30);
