function [F obj] = SpectralClustering_norm21(A, D, class_num)
% Unsupervised Learning using L21-norm minimization
% min_F \sum_ij A_ij*||Fi-Fj||_2

% A: similarity matrix on graph, n*n matrix
% D: identity matrix or degree matrix, n*n matrix
% F: embedded result, n*c matrix
% obj: objective values during the iterations

% Feiping Nie, Hua Wang, Heng Huang, Chris Ding.
% Unsupervised and Semi-supervised Learning via L1-norm Graph.
% The 13th International Conference on Computer Vision (ICCV), Barcelona, Spain, 2011.


obj = zeros(20,1);

[v d] = eig(A, D);
d = diag(d);
[d idx] = sort(d,'descend');
F = v(:,idx(1:class_num));
F = F*diag(sqrt(1./diag(F'*D*F)));
%n = size(A,1); F = orth(rand(n,class_num));

W = sqrt(abs(L2_distance_subfun(F',F'))+eps);
for iter = 1:20
    A1 = A./W;
    A1 = (A1+A1')/2;
    L1 = diag(sum(A1)) - A1;
    [v d] = eig(L1, D);
    d = diag(d);
    [d idx] = sort(d);
    F = v(:,idx(1:class_num));
    F = F*diag(sqrt(1./diag(F'*D*F)));
    W = sqrt(abs(L2_distance_subfun(F',F'))+eps);
    
    obj(iter) = sum(sum((A.*W)));    
end;



% compute squared Euclidean distance
% ||A-B||^2 = ||A||^2 + ||B||^2 - 2*A'*B
function d = L2_distance_subfun(a,b)
% a,b: two matrices. each column is a data
% d:   distance matrix of a and b


if (size(a,1) == 1)
  a = [a; zeros(1,size(a,2))]; 
  b = [b; zeros(1,size(b,2))]; 
end

aa=sum(a.*a); bb=sum(b.*b); ab=a'*b; 
d = repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab;
d = real(d);