function result = combine_kernels(weights, kernels)
    % length of weights should be equal to length of matrices
    n = length(weights);
    result = zeros(size(kernels(:,:,1)));    
    
    for i=1:n
        result = result + weights(i) * kernels(:,:,i);
end

function [data,rows,cols]=loadtabfile(filename)
	f = fopen(filename,'rt');
	
	line = fgetl(f);
	cols = textscan(line,'%s');
	cols = cols{1,1}; % first column for row labels
	
	rows = cell(0);
	data = zeros(0);
	while ~feof(f)
		line = fgetl(f);
		if numel(line) <= 1, break; end
		pos = find(line==9,1);
		rows = cat(1,rows,{line(1:pos-1)});
		data = cat(1,data,sscanf(line(pos:end),'%f')');
	end
	
	fclose(f);
end

function k = kernel_RBF(X,Y,gamma)
	r2 = repmat( sum(X.^2,2), 1, size(Y,1) ) ...
	+ repmat( sum(Y.^2,2), 1, size(X,1) )' ...
	- 2*X*Y' ;
	k = exp(-r2*gamma); 
end

function d = kernel_gip_0(adjmat,dim, gamma)
    y = adjmat;

    % Graph based kernel
	if dim == 1
        d = kernel_RBF(y,y,gamma);
    else
        d = kernel_RBF(y',y',gamma);
    end

	
	d(d==1)=0;
	d(logical(eye(size(d,1))))=1;
	
end

function New_x=remove_zeros(X)
	val_index = [];
	New_x = [];
	for i=1:size(X,2)
		a = X(:,i);
		val_a = var(a);
		val_index = [val_index,val_a];
		if val_a~=0
			New_x = [New_x,a];
		end
	end
	
end

function cv_index = crossvali_func(y,nfolds)

cv_index = zeros(size(y,1),1);
step =floor(size(y,1)/nfolds);
p=randperm(size(y,1));

for j=1:nfolds

	if j~=nfolds
		st=(j-1)*step+1;
		sed=(j)*step;

	else
		st=(j-1)*step+1;
		sed=size(y,1);
	end
		cv_p=[st:sed];
		ix = p(cv_p);
		cv_index(ix) = j;
end

end

function AUPR=calculate_aupr(celllines,predicts)
	if nargin > 1
		[~,i] = sort(predicts,'descend');
		celllines = celllines(i);
	end
	
	cumsums = cumsum(celllines)./reshape(1:numel(celllines),size(celllines));
	AUPR = sum(cumsums(~~celllines));
	pos = sum(celllines);
	AUPR = AUPR / pos;
end

function AUC=calculate_auc(celllines,predicts)
	if nargin > 1
		[~,i] = sort(predicts,'descend');
		celllines = celllines(i);
	end
	
	cumsums = cumsum(celllines);
	AUC = sum(cumsums(~celllines));
	pos = sum(celllines);
	neg = sum(~celllines);
	if pos == 0, warning('Calculate auc: no positive celllines'); end
	if neg == 0, warning('Calculate auc: no negative celllines'); end
	AUC = AUC / (pos * neg + eps);
end

function [ data ] = process_kernel( data )
    k = (data+data')/2;
    e = max(0, -min(eig(data)) + 1e-4);
    data = k + e*eye(length(data));
end

function S=Knormalized(K)
K = abs(K);
kk = K(:);
kk(find(kk==0)) = [];
min_v = min(kk);
K(find(K==0))=min_v;

D=diag(K);
D=sqrt(D);
S=K./(D*D');

end

function [LapA,beta_1,beta_2] = mknmf(Klist_1,Klist_2,y,lamda_T,lamda_L,lamda_1,lamda_2,lamda_w,k1,k2,interMax,isPro) 
LapA=[];
KK=[];
[n,m] = size(y);
number_k1 = size(Klist_1,3);
beta_1 = ones(number_k1,1)/number_k1;
number_k2 = size(Klist_2,3);
beta_2 = ones(number_k2,1)/number_k2;

phi_1 = zeros(number_k1,number_k1);
phi_2 = zeros(number_k2,number_k2);

for ii=1:number_k1
	for jj=1:number_k1
		mm1 = Klist_1(:,:,ii);
		mm2 = Klist_1(:,:,jj);
		phi_1(ii,jj) =  trace(mm1*mm2');
	end
end

for ii=1:number_k2
	for jj=1:number_k2
		mm1 = Klist_2(:,:,ii);
		mm2 = Klist_2(:,:,jj);
		phi_2(ii,jj) =  trace(mm1*mm2');
	end
end

l_1=ones(number_k1,1);
l_2=ones(number_k2,1);

zeta_1 = [];
zeta_2 = [];

	W1 = combine_kernels(beta_1, Klist_1);
	W2 = combine_kernels(beta_2, Klist_2);
if isPro==1
y = preprocess_Y(y,W1,W2,5,0.7);
end

[U1,S_k,V1] = svds(W1,k1);
G1 = U1*(S_k^0.5);  


[U2,S_k,V2] = svds(W2,k2);
G2 = U2*(S_k^0.5); 

A = G1;
B = G2;
k_r1 = eye(k1);
k_r2 = eye(k2);
Theta = zeros(k1,k2);

for o=1:interMax

inv_BB = pinv(B'*B);
t_inv_B = pinv(B');


	a = A'*A;
	b = lamda_T*inv_BB; 
	c = A'*y*t_inv_B;
	Theta = sylvester(a,b,c);
	
	A = (y*B*Theta' + lamda_1*W1*A)/(Theta*B'*B*Theta' + lamda_L*k_r1 + lamda_1*A'*A);
	
	B = (y'*A*Theta + lamda_2*W2*B)/(Theta'*A'*A*Theta + lamda_L*k_r2 + lamda_2*B'*B);
	
	
	zeta_1 = computer_zeta(A,Klist_1);
	temp_1=[];temp_2=[];
	temp_1 = (l_1'*((phi_1 + lamda_w*eye(number_k1))\zeta_1)) - 1;
	temp_2 = (l_1'*((phi_1 + lamda_w*eye(number_k1))\l_1));
	beta_1 = (phi_1 + lamda_w*eye(number_k1))\(zeta_1 - (temp_1/temp_2)*l_1);
	
	
	
	zeta_2 = computer_zeta(B,Klist_2);
	temp_1=[];temp_2=[];
	temp_1 = (l_2'*((phi_2 + lamda_w*eye(number_k2))\zeta_2)) - 1;
	temp_2 = (l_2'*((phi_2 + lamda_w*eye(number_k2))\l_2));
	beta_2 = (phi_2 + lamda_w*eye(number_k2))\(zeta_2 - (temp_1/temp_2)*l_2);
	
	
	W1 = combine_kernels(beta_1, Klist_1);
	W2 = combine_kernels(beta_2, Klist_2);
	

	
end
	
	
LapA = A*Theta*B';

end


function zeta_vector = computer_zeta(AB,KK)

	zeta_vector = zeros(size(KK,3),1);

	for ss=1:size(KK,3)
		zeta_vector(ss) = trace(AB'*KK(:,:,ss)*AB);
	end


end


function obj_v = computing_err(y,A,B,S1,S2,Theta)
		obj_1 = y-A*Theta*B';
	
	obj_2 = S1-A*A';
	obj_3 = S2-B*B';
	obj_v = norm(obj_1,'fro') + norm(Theta,'fro') + norm(A,'fro') + norm(B,'fro') + norm(obj_2,'fro') +norm(obj_3,'fro');

end


function Y=preprocess_Y(Y,Sd,St,K,eta)

    eta = eta .^ (0:K-1);

    y2_new1 = zeros(size(Y));
    y2_new2 = zeros(size(Y));

    empty_rows = find(any(Y,2) == 0);   % get indices of empty rows
    empty_cols = find(any(Y)   == 0);   % get indices of empty columns

    % for each drug pair i...
    for i=1:length(Sd)
        drug_sim = Sd(i,:); % get similarities of drug i to other drugs
        drug_sim(i) = 0;    % set self-similiraty to ZERO

        indices  = 1:length(Sd);    % ignore similarities 
        drug_sim(empty_rows) = [];  % to drugs of 
        indices(empty_rows) = [];   % empty rows

        [~,indx] = sort(drug_sim,'descend');    % sort descendingly
        indx = indx(1:K);       % keep only similarities of K nearest neighbors
        indx = indices(indx);   % and their indices

        drug_sim = Sd(i,:);
        y2_new1(i,:) = (eta .* drug_sim(indx)) * Y(indx,:) ./ sum(drug_sim(indx));
    end

    % for each cellline j...
    for j=1:length(St)
        cellline_sim = St(j,:); % get similarities of cellline j to other celllines
        cellline_sim(j) = 0;    % set self-similiraty to ZERO

        indices  = 1:length(St);        % ignore similarities 
        cellline_sim(empty_cols) = [];    % to celllines of
        indices(empty_cols) = [];       % empty columns

        [~,indx] = sort(cellline_sim,'descend');  % sort descendingly
        indx = indx(1:K);       % keep only similarities of K nearest neighbors
        indx = indices(indx);   % and their indices


        cellline_sim = St(j,:);
        y2_new2(:,j) = Y(:,indx) * (eta .* cellline_sim(indx))' ./ sum(cellline_sim(indx));
    end


    Y = max(Y,(y2_new1 + y2_new2)/2);

end

clear
seed = 1234;
rand('seed', seed);
nfolds = 5; nruns=1;

[y,l1,l2] = loadtabfile(['D:/MKNMF/adjmat.txt']);

gamma=0.5;
gamma_fp = 4;

fold_aupr_MKNMF_ka=[];fold_auc_MKNMF_ka=[];

preW = 1;
k1 = 10;k2 = 10;
lambda1 = 2^-0;lambda2=2^-0;
lamda_T=2^-0;lamda_L=2^-0;lamda_w = 2^-0;
interMax = 10;
globa_true_y_lp=[];
globa_predict_y_lp=[];
for run=1:nruns
    
     crossval_idx = crossvali_func(y(:),nfolds);

    for fold=1:nfolds
        t1 = clock;
        train_idx = find(crossval_idx~=fold);
        test_idx  = find(crossval_idx==fold);

        y_train = y;
        y_train(test_idx) = 0;

       
        k1_paths = {['D:/MKNMF/CDNA_CV.txt'],...
          ['D:/MKNMF/CGE_CV.txt'],...
          ['D:/MKNMF/CRNA.txt'],...
          ['D:/MKNMF/CNMF.txt'],...
          };
    K1 = [];
    for i=1:length(k1_paths)
      [mat, labels] = loadtabfile(k1_paths{i});
      mat = process_kernel(mat);
      K1(:,:,i) = Knormalized(mat);
    end
    
    k2_paths = {['D:/MKNMF/DFP_filter.txt'],...
          ['D:/MKNMF/DSE_filter.txt'],...
          ['D:/MKNMF/DTI_filter.txt'],...
          ['D:/MKNMF/DNMF.txt'],...
          };
    K2 = [];
    for i=1:length(k2_paths)
      [mat, labels] = loadtabfile(k2_paths{i});
      mat = process_kernel(mat);
      K2(:,:,i) = Knormalized(mat);
    end
    
		
        

		[A_cos_com,beta_1,beta_2] = mknmf(K1,K2,y_train,lamda_T,lamda_L,lambda1,lambda2,lamda_w,k1,k2,interMax,preW);beta_1
		beta_2
		
		t2=clock;
		etime(t2,t1)
		
		
        yy=y;
		test_labels = yy(test_idx);
		predict_scores = A_cos_com(test_idx);
		
		aupr_MKNMF_A_KA=calculate_aupr(test_labels,predict_scores);
		
    AUC_MKNMF_KA=calculate_auc(test_labels,predict_scores);
		
		fprintf('---------------\nRUN %d - FOLD %d  \n', run, fold)

		fprintf('%d - FOLD %d - weighted_kernels_MKNMF_AUPR: %f \n', run, fold, aupr_MKNMF_A_KA)
		

		fold_aupr_MKNMF_ka=[fold_aupr_MKNMF_ka;aupr_MKNMF_A_KA];
		fold_auc_MKNMF_ka=[fold_auc_MKNMF_ka;AUC_MKTNF_KA];

		
		globa_true_y_lp=[globa_true_y_lp;test_labels];
		globa_predict_y_lp=[globa_predict_y_lp;predict_scores];
		%break;
		
    end
    
    
end
RMSE = sqrt(sum((globa_predict_y_lp-globa_true_y_lp).^2)/length(globa_predict_y_lp))



mean_aupr_kronls_ka = mean(fold_aupr_MKNMF_ka)
mean_auc_kronls_ka = mean(fold_auc_MKNMF_ka)
