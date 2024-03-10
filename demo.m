clear;
clc;


%data_path = 'new path';

% take Handwritten for an example
data_name = '3';
fprintf('\ndata_name: %s', data_name);

%% pre-process, all algs do so
load([data_name, '.mat']);
        
k = length(unique(Y));
V = length(X);
for v = 1 : V    
    X{v} = mapstd(X{v},0,1)';
    XX{v} = X{v}*X{v}';
end

for i =1:length(X)
    m_bar{i}= size(X{i},1);
end
%% para setting
m = k;
%m_bar = k;
%d_bar = size(X_bar, 1);
ii = 1;
% lambda1=[1e2];
% lambda2=[1e-2];
% lambda3=[1e2];
lambda1=[1e0,1e1,1e2,1e3];
lambda2=[1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3];
lambda3=[1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3];
% lambda1=[1e1];
%  lambda2=[1e2];
% lambda3=[1e3];
%for maxiter = 20
for i = 1:length(lambda1)
   for j = 1:length(lambda2)
      for h = 1:length(lambda3)
          fprintf('\nii = %d;', ii);
          tic;
          [UU,W,A,Z,P,obj] = algo_p(X',Y, lambda1(i),lambda2(j),lambda3(h),m_bar, m,XX);
          %[UU, D, A, Z, Z_bar, Q, obj,times] = algo_qp(X, X_bar, Y, tau, lambda1, lambda2, d_bar, m_bar, m, maxiter); % X,Y,lambda,d,numanchor
          res = myNMIACCwithmean(UU,Y,k); % [ACC nmi Purity Fscore Precision Recall AR Entropy]
          time = toc;
          %fprintf('\n lambda1:%.5f lambda3:%.5f \n ACC: %.4f NMI: %.4f PUR: %.4f F: %.4f ARI: %.4f Time: %.4f \n',[lambda1(i) lambda3(h) res(1) res(2) res(3) res(4) res(7) time]);                  
          fprintf('\n lambda1:%.5f lambda2:%.5f lambda3:%.5f \n ACC: %.4f NMI: %.4f PUR: %.4f F: %.4f ARI: %.4f Time: %.4f \n',[lambda1(i) lambda2(j) lambda3(h) res(1) res(2) res(3) res(4) res(7) time]);                       
      result(ii, :) = [res(1) res(2) res(3) res(4) res(7) time];
      ii = ii + 1; 
      end 
   end
end
%end 
aa=max(result(:,1))
% acc=[];
% for ii=1:size(resall,1)
%     for jj=1:size(resall,2)
%         acc(ii,jj)=resall{ii,jj}(1);
%         [hh]=max(max(acc));

%         [row,cell]=find(acc==hh);
%     end
% end
% maxres=resall{row,cell};%ACC nmi Purity Fscore Precision Recall AR Entropy

% ZZT = Z_bar' * Z_bar;
% imagesc(ZZT)
% colormap(gray(256))
%save("Fashion_MV.mat")
% ZZT = Z' * Z;
% imagesc(ZZT)
% colormap(gray(256))
%save('g_3.mat')
