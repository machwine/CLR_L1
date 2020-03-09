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


num=size(X,1);%������Ŀ
dim=size(X,2);%����ά��
wanted_class=length(unique(Y));%��Ҫ�ֵ����
k=wanted_class;%���
W = constructW_PKN(X', 5, 1);% 5�������ɵ�W %���������һ������ W��ֵ����0��1 W:nxn

Dw = zeros(num,num);
for i=1:num
    Dw(i,i) = sum(W(i,:));
end
Lw = Dw - W;
[F,firstKlemda] = firstKEigenVector(Lw,k);%��ʼ��F Lw��ǰkС��������

%ͨ��F��dij
d2=L2_distance_1(F',F');
d = sqrt(d2);

lamda=600;%landa

S = zeros(size(W));
loss=1000000; %����loss�ܴ�
iteration=0;
Ds = zeros(num,num);
for j=1:110
    for i=1:num
        S(i,:) = EProjSimplex_new((W(i,:)-lamda/2*d(i,:)));%fix F get S
    end
    S = (S+S')/2;
   

    Ds = diag(sum(S));
%     for i=1:num %��S�ĶȾ���
%         Ds(i,i) = sum(S(i,:));
%     end

    [F ,~] = SpectralClustering_norm21(S, Ds, k);%fix S get F
    
    %�����
    loss1 = (norm(S-W,'fro').^2);
    loss2 = sum(sum(triu(S.*d)));%�����Ǿ���
    if abs(loss-(loss1+loss2)) < 10^-8
        break;
    else
        loss = loss1 + loss2;
    end
    iteration=iteration+1; 
    d2=L2_distance_1(F',F');
    d = sqrt(d2);
    
    
end

[clusternum, predict]=graphconncomp(sparse(S)); predict = predict';%����Ԥ����
 result = ClusteringMeasure(Y, predict)
 nmi_result = nmi(Y', predict')

