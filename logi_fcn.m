function logi_fcn
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
% for perr = 5:35
% resSubPath = ['perr',num2str(perr-4)];
perr = 35;
% may changed in the following function
min_mu = 0.5; % set the smallest strongly convex parameter mu in S
max_Lips = 1; % set the Lipschitz constant

% generate the network W
W = cell(1,perr);
rng('shuffle')
perm = randperm(n,perr);
for k = 1:perr
    per = perm(k)/L;
    W{k} = generateW(L,per);
end

% W{1} = generateW(L,15/L);
% W{2} = generateW(L,35/L);

W_num = length(W);

% generate the smooth function S
[M, x_ori, y_ori] = generateS_logi(m, p, n,...
    min_mu,max_Lips);

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
iter    = 200000;      % the maximum number of iterations
tol     = 1e-11;     % tolerance, this controls |x-x_star|_F, not divided by |x_star|_F
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

methods = {'NIDSS'};
LineSpecs = {'-g','-.b',':m'};
numMethods = length(methods);
outputs = cell(numMethods,1);
legend_lab = cell(numMethods,1);
% call methods
for i = 1:numMethods
    paras.method = methods{i};
    
    switch methods{i}
        case {'DIGing-ATC'}
            paras.alpha = cRate./max_Lips*ones(n,1);
            paras.atc = 1;
            outputs{i}  = obj.minimize_DIGing(paras);
            legend_lab{i} = 'DIGing-ATC';
            
        case {'DIGing'}
            paras.alpha = cRate./max_Lips*ones(n,1);
            paras.atc = 0;
            outputs{i} = obj.minimize_DIGing(paras);
            legend_lab{i} = 'DIGing';
            
        case {'NIDSS-adaptive'}
            cRate = 1; % rate of the step size < 2
            alpha = cRate./Lips;
            paras.alpha = alpha;
            
            %
            eye_L = eye(n);
            c = zeros(1,W_num);
            for j = 1:W_num
                I_W = eye_L-W{j};
                [U,S,V] = svd(I_W);
                a = diag(S);
                inv_I_W = U*diag([a(1:end-1).^(-1);0])*V';
                alpha2 = diag(sqrt(1./alpha));
                eigs = eig(alpha2*inv_I_W*alpha2);
                [~, indmin]=min(eigs);
                ind = ones(size(eigs)); ind(indmin) = 0;
                lambda_n_1 = min(eigs(logical(ind)));
                c(j) = lambda_n_1;
            end
            
            paras.c = c;
            outputs{i} = obj.minimize(paras);
            legend_lab{i} = 'NIDS-adaptive';
            
        case {'NIDSS'}
            cRate = 2; % rate of the step size < 2
            
            alpha = cRate./max_Lips*ones(n,1);
            c = 1./(1-lambdan)/max(alpha);
            paras.alpha = alpha;
            paras.c = c;
            paras.forcetTildeW = 0;
            outputs{i} = obj.minimize(paras);
            legend_lab{i} = 'NIDS';
            
        case {'NIDSS-F'}
            cRate = 1; % rate of the step size < 2
            
            alpha = cRate./max_Lips*ones(n,1);
            
            paras.alpha = alpha;
            
            paras.forcetTildeW = true;
            paras.method = 'NIDSS';
            
            outputs{i} = obj.minimize(paras);
            legend_lab{i} = 'NIDS-$c={1/( 2\alpha)}$';
            
        case {'EXTRA'}
            cRate = 1;
            
            paras.alpha = cRate./max_Lips*ones(n,1);
            outputs{i} = obj.minimize(paras);
            legend_lab{i} = 'EXTRA';
        otherwise
            disp('????')
    end
end
for i = 1:numMethods
    semilogy(outputs{i}.err/norm_x_star,LineSpecs{i});
    hold on;
end

xlabel('number of iterations');
ylabel('$\frac{\left\Vert \mathbf{x}-\mathbf{x}^{*}\right\Vert}{\left\Vert \mathbf{x}^{*}\right\Vert}$','FontSize',20,'Interpreter','LaTex');
% title('Quadratic Function','FontSize',12)
legend(legend_lab,'FontSize',10,'Interpreter','LaTex');
% saveas(h,[resSubPath,'.png']);
% xlim([0 80])
%     close;
prob.M = M;
prob.x_ori = x_ori;
prob.y_ori = y_ori;
prob.lam = lam;
prob.W = W;

end

function a = funGradS(x)
global n p M y_ori
a = zeros(n, p);
for j = 1:n
    eta = M(:,:,j) * (x(j,:))';
    eta = 1 ./ (1+exp(-eta)) - y_ori(:,j);
    a(j,:) = mean(diag(eta) * M(:,:,j),1);
end
end

function a = funS(x)
global n M y_ori
a = 0;
for j = 1:n
    eta = M(:,:,j) * (x(j,:))';
    eta = log(1+exp(-eta)) .* y_ori(:,j) + log(1+exp(eta)) .* (1 - y_ori(:,j));
    a = a + mean(eta,1);
end
end
