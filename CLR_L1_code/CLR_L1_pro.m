clear all;
% load('dig1-10_uni.mat');
% load('uspst_uni.mat');
% load('wine_uni.mat');
% load('COIL20.mat');
% load('col20.mat');
% load('AR.mat');
% X=fea;
% Y=gnd;
load('yeast_uni10.mat');  % load


num=size(X,1);%样本数目
dim=size(X,2);%样本维数
wanted_class=length(unique(Y));%想要分的类别
k=wanted_class;%类别
W = constructW_PKN(X', 5, 1);% 5近邻生成的W %这里出现了一个问题 W的值不是0或1 W:nxn

Dw = zeros(num,num);
for i=1:num
    Dw(i,i) = sum(W(i,:));
end
Lw = Dw - W;
[F,firstKlemda] = firstKEigenVector(Lw,k);%初始化F Lw的前k小特征向量

%通过F求dij
d2=L2_distance_1(F',F');
d = sqrt(d2);

lamda=600;%landa

S = zeros(size(W));
loss=1000000; %假设loss很大
iteration=0;
Ds = zeros(num,num);
for j=1:110
    for i=1:num
        S(i,:) = EProjSimplex_new((W(i,:)-lamda/2*d(i,:)));%fix F get S
    end
    S = (S+S')/2;
   

    Ds = diag(sum(S));
%     for i=1:num %求S的度矩阵
%         Ds(i,i) = sum(S(i,:));
%     end

    [F ,~] = SpectralClustering_norm21(S, Ds, k);%fix S get F
    
    %求误差
    loss1 = (norm(S-W,'fro').^2);
    loss2 = sum(sum(triu(S.*d)));%上三角矩阵
    if abs(loss-(loss1+loss2)) < 10^-8
        break;
    else
        loss = loss1 + loss2;
    end
    iteration=iteration+1; 
    d2=L2_distance_1(F',F');
    d = sqrt(d2);
    
    
end

[clusternum, predict]=graphconncomp(sparse(S)); predict = predict';%分类预测结果
 result = ClusteringMeasure(Y, predict)
 nmi_result = nmi(Y', predict')

