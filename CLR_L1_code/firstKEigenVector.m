function [firstkVector,firstKlemda]=firstKEigenVector(A,k)
%输入一个方阵
%将特征值和特征向量重排序，按照特征值升序来排
%取前k个特征值和对应的特征向量

%测试用
%A=magic(5);

matrixSize=size(A);
[v,d] = eig(A) ;
v = [v ; diag(d)' ; ]' ;%将特征值的信息加入到矩阵便于按特征值排序
v = sortrows(v,matrixSize+1) ;%升序排列
firstKlemda = v(1:k,matrixSize(1)+1) ;
firstkVector = v(1:k,1:matrixSize)';