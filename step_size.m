function step_size
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      minimize S(x)   subject to Wx = x
%    S is differentiable: S = 1/2||Mx-y||_2^2
%    W is the given mixing matrix

%    Reference: A Decentralized Proximal-Gradient Method with Network
%               Independent Step-zsizes and Seperated Convergence Rates
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global n m p M y_ori lam
path(path, '.\fcns')
n = 40; % number of nodes
m = 60;
p = 50; % the dimension of x on each nodes

L = n;
% per_set=[14,18]; % parameters control the connectivity of the network
% for perr = per_set
%     per = perr/L;
% resSubPath = ['per','1-40','overL_mu0_5'];

% may changed in the following function
min_mu = 0.5; % set the smallest strongly convex parameter mu in S
max_Lips = 1; % set the Lipschitz constant

% generate the network W
W = cell(1,35);
for k = 1:35
    per = k/L;
    W{k} = generateW(L, per);
end

% W{1} = generateW(L,15/L);
% W{2} = generateW(L,35/L);

W_num = length(W);

% generate the smooth function S
[M, x_ori, y_ori] = generateS(m, p, n,...
    'withoutNonsmoothR',min_mu,max_Lips);

rng('shuffle')

% find the smallest eigenvalue of W
lambdan = zeros(1,W_num);
for i = 1:W_num
    [~, lambdan(i)] = eigW(W{i});
end

% find the Lipschitz constants and the strongly convex parameters of
% S_i
[Lips,mus] = getBetaSmoothAlphaStrong;
max_Lips   = max(Lips);
min_mu     = min(mus);

% set parameters
iter    = 1000;      % the maximum number of iterations
tol     = 1e-6;     % tolerance, this controls |x-x_star|_F, not divided by |x_star|_F
x0      = zeros(n,p);% initial guess of the solution
x_star  = x_ori;     % true solution
% Set the parameter for the solver
paras.min_mu    = min_mu;
paras.max_Lips  = max_Lips;
paras.x_star    = x_star;
paras.n         = n;    % the number of nodes
paras.p         = p;    % the dimension of x on each nodes
paras.iter      = iter; % max iteration
paras.x0        = x0;   % the initial x
paras.W         = W;    % the mixing matrix
paras.tol       = tol;  % tolerance

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% start using the NIDS class
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
obj            =  NIDS;  % using the class PrimalDual

obj.getS       = @(x) feval(@funS, x);
obj.getGradS   = @(x) feval(@funGradS, x);

h = figure;
set(h, 'DefaultLineLineWidth', 4)
norm_x_star = norm(x_star, 'fro');

LineSpecs = {':b','-.k','-g','-.r','--m'};

cRate = 1; % rate of the step size < 2


numcRate = length(cRate);
legend_lab = cell(numcRate,1);
paras.method = 'NIDSS';
% num_iter = zeros(10,numcRate);

for i = 1:numcRate
    alpha = cRate(i)./max_Lips*ones(n,1);
    c = 1./(1-lambdan)/max(alpha);
    paras.alpha = alpha;
    paras.c = c;
    paras.forcetTildeW = 0;

    outputs = obj.minimize(paras);


    
%     disp(i)
    legend_lab{i} = ['NIDS-',num2str(cRate(i)),'/L'];
    semilogy(outputs.err/norm_x_star,LineSpecs{i});
    hold on;
end

% x = mean(num_iter,1);
% save('mean.mat','x');
xlabel('number of iterations');
ylabel('$\frac{\left\Vert \mathbf{x}-\mathbf{x}^{*}\right\Vert}{\left\Vert \mathbf{x}^{*}\right\Vert}$','FontSize',20,'Interpreter','LaTex');
title('Smooth Function')
legend(legend_lab,'FontSize',10,'Interpreter','LaTex');
% saveas(h,[resSubPath,'_compa3.fig']);
% xlim([0 80])
%     close;
% prob.M = M;
% prob.x_ori = x_ori;
% prob.y_ori = y_ori;
% prob.lam = lam;
% prob.W = W;

% save([resSubPath,'_compa3_prob.mat'],'prob');

end

function a = funGradS(x)
global n p M y_ori
a = zeros(n, p);
for j = 1:n
    a(j,:) = (M(:,:,j)' * (M(:,:,j) * (x(j,:))' - y_ori(:,j)))';
end
end

function a = funS(x)
global n M y_ori
a = 0;
for j = 1:n
    a   = a + 0.5 * sum((M(:,:,j) * (x(j,:))' - y_ori(:,j)).^2);
end
end
