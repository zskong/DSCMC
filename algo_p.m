function [UU,W,A,Z,P,obj] = algo_p(X,Y, lambda1, lambda2,lambda3,m_bar, m, XX)
% m      : the number of anchor. the size of Z is m*n.
% lambda : the hyper-parameter of regularization term.
% X      : n*di
% 对比组：其中没有权重学习
%% initialize
maxiter = 20 ; % the number of iterations
numview = length(X);
numsample = size(Y, 1);

Z = zeros(m, numsample);
Z(:, 1 : m) = eye(m);
A = zeros(m, m);
for i = 1:numview
   d(i) = size(X{i}, 1);
   W{i} = zeros(d(i), m);
   P{i} = zeros(m, d(i));
end

opt.disp = 0;
%alpha = ones(1,numview)/numview;
flag = 1;
iter = 0;
%%
fprintf(' iter =');
while flag
    iter = iter + 1;
    fprintf(' %d', iter)
    
    %% optimize W
    %parfor i = 1 : numview
    %sumAlpha = 0;
    for i = 1 : numview
        %al2 = alpha(i)^2;
        %sumAlpha = sumAlpha + al2;
        [U, ~, V] = svd(X{i} * Z' * A', 'econ');
        W{i} = U * V';
    end
    clear U V
    
    %% optimize A
    temp = zeros(m, m);
    for i = 1 : numview
        temp = temp +  W{i}' * X{i} * Z'+ lambda1 * P{i} * X{i} * Z';
    end
    [U, ~, V] = svd(temp, 'econ');
    A = U * V';
    clear U V temp
  
    %% optimize P
    %v  = sqrt(sum(Q .* Q, 1) + eps);
  %  vv  = diag(1 ./ (v));
  %  Q = (lambda1 * Z * Z_bar') / (lambda1 * Z_bar * Z_bar' + lambda2 * vv);
  %  clear v vv   
  for i =1:length(X)
      orl{i} = lambda1 * XX{i} + lambda3 * eye(m_bar{i});
      P{i} =  (lambda1 * A * Z * X{i}') / orl{i} ;
  end
  %  clear v vv              
    %% optimize Z

    H = 2 * (numview + lambda1 + lambda2) * eye(m);
    H = (H + H') / 2;
    options = optimset( 'Algorithm','interior-point-convex','Display','off'); % Algorithm é»è®€äž? interior-point-convex
    for j = 1 : numsample
        ff = 0;
        for i = 1 : numview
          ff =  ff - 2 * X{i}(:, j)' * W{i} * A - 2 * lambda1 * X{i}(:, j)' * P{i}' * A;
        end
        Z(:, j) = quadprog(H, ff',[],[],ones(1, m),1,zeros(m, 1),ones(m, 1),[], options);
    end 
    %  %% optimize alpha
    %  M = zeros(numview,1);
    % for iv = 1:numview
    %     M(iv) = norm(X{iv} - W{iv} * A * Z,'fro')^2;
    % end
    % Mfra = M.^-1;
    % QQ = 1/sum(Mfra);
    % alpha = QQ*Mfra;



        
    %% calculate function value and check convergence
    term1 = 0; term2 = 0; term4 = 0;
    for i = 1 : numview
       term1 = term1 + norm(X{i} - W{i} * A * Z, 'fro') ^ 2;
       term2 = term2 + norm(P{i} * X{i} - A * Z, 'fro') ^ 2;
       term4 = term4 + sum(sqrt(sum(P{i} .* P{i}, 1)));
    end
    term3 = norm(Z, 'fro') ^ 2;
    %obj(iter) = term1+ lambda1 * term2+ lambda2*term3 + lambda3*term4;
    %obj(iter) = numview \ term1 +  lambda1 * (numview \ term2) + lambda2 * term3 + lambda3 * (numview \ term4);
    obj(iter) = term1+ lambda1 * term2 + lambda2*term3 + lambda3*term4;
    %obj(iter) = term1+ lambda1 * term2 + lambda3*term4;
    %if (iter>9) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-3 || iter>maxiter || obj(iter) < 1e-10)
    if iter>=maxiter
        [UU, ~, ~]=svd(Z', 'econ');
        flag = 0;
    end
end
         
         
    
