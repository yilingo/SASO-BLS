%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% Comparations among SA methods%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2022

clear;
clc;
close all; 
warning off all;
format compact;

%% add path
addpath('utils');
addpath('functions');

my_dir = pwd ; 
cd(my_dir)
addpath(genpath(my_dir))

%% DataSet Load 
DataSet = 'TE'; 
sigfun = 'logsig';
NormMethod = 4;
[train_x,test_x,ForTrain_y,ForTest_y] = DataSetLoad.load(DataSet);

%% data norm
[ForTrain_x,ForTest_x] = DataSetLoad.Norm(train_x,test_x,NormMethod);

%% Parameter set 
% model parameters
NumPerWin = 10;  % Nodes number of the feature mapping layer per window
NumWindow = 10;  % Nodes number of windows of the feature mapping layer
NumEnhPer = 100; % Nodes number of enhancement layer added per test

% incremental learning: None
NumFeaPerInc = [];
NumEnhRelPerInc = [];
NumEnhPerInc = [];

% other parameters
L2Param = 2^-30; %L2 parameter
ShrScale = .8;   %the l2 regularization parameter and the shrinkage scale of the enhancement nodes   
BanType = 'All'; %FeatureNodes %All %EnhanNodes
StartStep = 0;
BanIndex = [];
ThetaSel = 0.085;
InitMed = 'GuassX'; %MeanX , GuassX, MeanHe,GuassHe
ifplot = false;
test_step = 10;
Repet_time = 10;

compared_method = {'FSA','TSA','SVSA','GVSA','ESA'}; 

for z = 1:Repet_time

    disp(['********Start the ', num2str(z), '-th round ********']);

    for i = 1:test_step
        % different number of Enhancement layer
        NumEnhance = i*NumEnhPer; 
        disp(['********Start the ', num2str(z), '-th round with ', num2str(NumEnhance), ' enhance nodes learning process********']);      
             

        % Model Initialization
        Model = SASO_Class(NumPerWin,NumWindow,NumEnhance,NumFeaPerInc,NumEnhRelPerInc,NumEnhPerInc,ShrScale,L2Param,BanIndex,StartStep,sigfun,InitMed);
    
        %% BLS training
        tic;
        Model = Model.Train(ForTrain_x,ForTrain_y); 
        BLS_time(z,i) = toc;
    
        % get BLS results
        TrainResult = Model.GetOutput(ForTrain_x);    
        TrainLabelDis = MyClassTools.ClassResult(ForTrain_y);
        ValResult = Model.GetOutput(ForTest_x);
        ValResultDis = MyClassTools.ClassResult(ValResult);
        ValLabelDis = MyClassTools.ClassResult(ForTest_y);
        ValIndex = Evaluation_idx(ValResultDis,ValLabelDis);
        [~,~,~,~,~,~,WMacro_P,WMacro_R,WMacro_F1] = ValIndex.Macro();
        BLSpara = ((length(ForTrain_x(1,:))+1)*NumPerWin*NumWindow+...
            (NumPerWin*NumWindow+1)*NumEnhance+(NumPerWin*NumWindow+NumEnhance+1)*10)/1000;
        
        % BLS output and save
        disp(['The Precision of BLS is ' ,num2str(WMacro_P)]);
        fprintf(2,['The Recall of BLS is ' ,num2str(WMacro_R),'\n']);
        fprintf(2,['The macro-F1 of BLS is ' ,num2str(WMacro_F1),'\n']);
        disp(['The parameter of BLS is ' ,num2str(BLSpara)]);
        BLS_Pre(z,i) = WMacro_P;
        BLS_Rec(z,i) = WMacro_R;
        BLS_F1(z,i) = WMacro_F1;
        BLS_Par(z,i) = BLSpara;


        
        %% SA preparation
        NumEachLabel = tabulate(TrainLabelDis);
        NumEech4SA = min(NumEachLabel(:,2));
        SelTrainA = Model.A_Matrix_Train;

        %% EET-SA for compression 
        if any(strcmpi('ESA',compared_method))
            tic;
            Model_ESA = OTAT_Off.OT_SA(Model,SelTrainA,NumEech4SA,'EET_SA');
            BLSESA_time(z,i) = toc;
    
            [Model_ESA,~] = Model_ESA.PrunOutput(ForTrain_x,BanType,ForTrain_y,'update');
            [~,ESAValResult] = Model_ESA.PrunOutput(ForTest_x,BanType,ForTest_y,'test');
            ESAValResultDis = MyClassTools.ClassResult(ESAValResult);
            ESAValIndex = Evaluation_idx(ESAValResultDis,ValLabelDis);
            [~,~,~,~,~,~,WMacro_P,WMacro_R,WMacro_F1]  = ESAValIndex.Macro();
    
            BanNumEnhanceESA = length(Model_ESA.BanNodes(Model_ESA.BanNodes>NumPerWin*NumWindow));
            BanNumFeatureESA = length(Model_ESA.BanNodes)-BanNumEnhanceESA;
            NumEnhanceESA = NumEnhance - BanNumEnhanceESA;
            NumFeatureESA = NumPerWin*NumWindow-BanNumFeatureESA;
            ESApara = ((length(ForTrain_x(1,:))+1)*NumFeatureESA+...
                (NumFeatureESA+1)*NumEnhanceESA+(NumFeatureESA+NumEnhanceESA+1)*10)/1000;  
        
            % ESA output and save
            BLSESA_Pre(z,i) = WMacro_P;
            BLSESA_Rec(z,i) = WMacro_R;
            BLSESA_F1(z,i) = WMacro_F1;
            BLSESA_Par(z,i) = ESApara;
            disp(['The Precision of ESA is --------------' ,num2str(WMacro_P)]);
            fprintf(2,['The Recall of ESA is --------------' ,num2str(WMacro_R),'\n']);
            fprintf(2,['The macro-F1 of ESA is --------------' ,num2str(WMacro_F1),'\n']);
            disp(['The parameter of ESA is --------------' ,num2str(ESApara)]);
        end

        %%  SV-SA for compression 
        if any(strcmpi('SVSA',compared_method))
            tic;
            Model_SVSA = OTAT_Off.OT_SA(Model,SelTrainA,NumEech4SA,'SV_SA');
            BLSSVSA_time(z,i) = toc;
    
            [Model_SVSA,~] = Model_SVSA.PrunOutput(ForTrain_x,BanType,ForTrain_y,'update');
            [~,SVSAValResult] = Model_SVSA.PrunOutput(ForTest_x,BanType,ForTest_y,'test');
            SVSAValResultDis = MyClassTools.ClassResult(SVSAValResult);
            SVSAValIndex = Evaluation_idx(SVSAValResultDis,ValLabelDis);
            [~,~,~,~,~,~,WMacro_P,WMacro_R,WMacro_F1]  = SVSAValIndex.Macro();
    
            BanNumEnhanceSVSA = length(Model_SVSA.BanNodes(Model_SVSA.BanNodes>NumPerWin*NumWindow));
            BanNumFeatureSVSA = length(Model_SVSA.BanNodes)-BanNumEnhanceSVSA;
            NumEnhanceSVSA = NumEnhance - BanNumEnhanceSVSA;
            NumFeatureSVSA = NumPerWin*NumWindow-BanNumFeatureSVSA;
            SVSApara = ((length(ForTrain_x(1,:))+1)*NumFeatureSVSA+...
                (NumFeatureSVSA+1)*NumEnhanceSVSA+(NumFeatureSVSA+NumEnhanceSVSA+1)*10)/1000;  
        
            % SVSA output and save
            BLSSVSA_Pre(z,i) = WMacro_P;
            BLSSVSA_Rec(z,i) = WMacro_R;
            BLSSVSA_F1(z,i) = WMacro_F1;
            BLSSVSA_Par(z,i) = SVSApara;
            disp(['The Precision of SVSA is --------------' ,num2str(WMacro_P)]);
            fprintf(2,['The Recall of SVSA is --------------' ,num2str(WMacro_R),'\n']);
            fprintf(2,['The macro-F1 of SVSA is --------------' ,num2str(WMacro_F1),'\n']);
            disp(['The parameter of SVSA is --------------' ,num2str(SVSApara)]);
        end

        %% GV-SA for compression 
        if any(strcmpi('GVSA',compared_method))
            tic;
            Model_GVSA = OTAT_Off.OT_SA(Model,SelTrainA,NumEech4SA,'GV_SA');
            BLSGVSA_time(z,i) = toc;
    
            [Model_GVSA,~] = Model_GVSA.PrunOutput(ForTrain_x,BanType,ForTrain_y,'update');
            [~,GVSAValResult] = Model_GVSA.PrunOutput(ForTest_x,BanType,ForTest_y,'test');
            GVSAValResultDis = MyClassTools.ClassResult(GVSAValResult);
            GVSAValIndex = Evaluation_idx(GVSAValResultDis,ValLabelDis);
            [~,~,~,~,~,~,WMacro_P,WMacro_R,WMacro_F1]  = GVSAValIndex.Macro();
    
            BanNumEnhanceGVSA = length(Model_GVSA.BanNodes(Model_GVSA.BanNodes>NumPerWin*NumWindow));
            BanNumFeatureGVSA = length(Model_GVSA.BanNodes)-BanNumEnhanceGVSA;
            NumEnhanceGVSA = NumEnhance - BanNumEnhanceGVSA;
            NumFeatureGVSA = NumPerWin*NumWindow-BanNumFeatureGVSA;
            GVSApara = ((length(ForTrain_x(1,:))+1)*NumFeatureGVSA+...
                (NumFeatureGVSA+1)*NumEnhanceGVSA+(NumFeatureGVSA+NumEnhanceGVSA+1)*10)/1000;  
        
            % GVSA output and save
            BLSGVSA_Pre(z,i) = WMacro_P;
            BLSGVSA_Rec(z,i) = WMacro_R;
            BLSGVSA_F1(z,i) = WMacro_F1;
            BLSGVSA_Par(z,i) = GVSApara;
            disp(['The Precision of GVSA is --------------' ,num2str(WMacro_P)]);
            fprintf(2,['The Recall of GVSA is --------------' ,num2str(WMacro_R),'\n']);
            fprintf(2,['The macro-F1 of GVSA is --------------' ,num2str(WMacro_F1),'\n']);
            disp(['The parameter of GVSA is --------------' ,num2str(GVSApara)]);
        end
 
        
        %% FPD-SA for compression
        if any(strcmpi('FSA',compared_method))
            tic;
            Model_FSA = FPD_SA_Off.SA(Model,SelTrainA,NumEech4SA,sigfun,ThetaSel); 
            BLSFSA_time(z,i) = toc;
        
            % get FPD-SA results
            [ModelFSA,~] = Model_FSA.PrunOutput(ForTrain_x,BanType,ForTrain_y,'update');  
            [~,FSAValResult] = ModelFSA.PrunOutput(ForTest_x,BanType,ForTest_y,'test');
            FSAValResultDis = MyClassTools.ClassResult(FSAValResult);
            FSAValIndex = Evaluation_idx(FSAValResultDis,ValLabelDis);
            [~,~,~,~,~,~,WMacro_P,WMacro_R,WMacro_F1]  = FSAValIndex.Macro();
        
            % get model param after FPD-SA 
            BanNumEnhanceFSA = length(Model_FSA.BanNodes(Model_FSA.BanNodes>NumPerWin*NumWindow));
            BanNumFeatureFSA = length(Model_FSA.BanNodes)-BanNumEnhanceFSA;
            NumEnhanceFSA = NumEnhance-BanNumEnhanceFSA;    
            NumFeatureFSA = NumPerWin*NumWindow-BanNumFeatureFSA;
            FSApara = ((length(ForTrain_x(1,:))+1)*NumFeatureFSA+...
                (NumFeatureFSA+1)*NumEnhanceFSA+(NumFeatureFSA+NumEnhanceFSA+1)*10)/1000;
            
            % FPD-SA output and save
            BLSFSA_Pre(z,i) = WMacro_P;
            BLSFSA_Rec(z,i) = WMacro_R;
            BLSFSA_F1(z,i) = WMacro_F1;
            BLSFSA_Par(z,i) = FSApara;
            disp(['The Precision of FSA is -------' ,num2str(WMacro_P)]);
            fprintf(2,['The Recall of FSA is -------' ,num2str(WMacro_R),'\n']);
            fprintf(2,['The macro-F1 of FSA is ' ,num2str(WMacro_F1),'\n']);
            disp(['The parameter of FSA is -------' ,num2str(FSApara)]);
        end
        
        %% traditional partial differential SA for compression
        if any(strcmpi('TSA',compared_method))
            tic
            Model_TSA = PD_TSA_Off.TSA(Model,SelTrainA,sigfun); 
            BLSTSA_time(z,i) = toc;
        
            % get TSA results
            [ModelTSA,FSATrainResult] = Model_TSA.PrunOutput(ForTrain_x,BanType,ForTrain_y,'update');
            [~,FSAValResult] = ModelTSA.PrunOutput(ForTest_x,BanType,ForTest_y,'test');
            TSAValResultDis = MyClassTools.ClassResult(FSAValResult);
            TSAValIndex = Evaluation_idx(TSAValResultDis,ValLabelDis);
            [~,~,~,~,~,~,WMacro_P,WMacro_R,WMacro_F1]  = TSAValIndex.Macro();
        
            % get model param after TSA 
            BanNumEnhanceTSA = length(Model_TSA.BanNodes(Model_TSA.BanNodes>NumPerWin*NumWindow));
            BanNumFeatureTSA = length(Model_TSA.BanNodes)-BanNumEnhanceTSA;
            NumEnhanceTSA = NumEnhance - BanNumEnhanceTSA;
            NumFeatureTSA = NumPerWin*NumWindow-BanNumFeatureTSA;
            TSApara = ((length(ForTrain_x(1,:))+1)*NumFeatureTSA+...
                (NumFeatureTSA+1)*NumEnhanceTSA+(NumFeatureTSA+NumEnhanceTSA+1)*10)/1000;  
        
            % TSA output and save
            BLSTSA_Pre(z,i) = WMacro_P;
            BLSTSA_Rec(z,i) = WMacro_R;
            BLSTSA_F1(z,i) = WMacro_F1;
            BLSTSA_Par(z,i) = TSApara;
            disp(['The Precision of TSA is --------------' ,num2str(WMacro_P)]);
            fprintf(2,['The Recall of TSA is --------------' ,num2str(WMacro_R),'\n']);
            fprintf(2,['The macro-F1 of TSA is --------------' ,num2str(WMacro_F1),'\n']);
            disp(['The parameter of TSA is --------------' ,num2str(TSApara)]);
        end

        %% plot table
        rank_list = [];
        % Define the table format
        formatSpec = '%.4f\t'; % specify format with 2 decimal places
        
        % Print the table header
        fprintf('method\tRecall\tPrecision\tmacro-F1\tpara\ttime\n')
        fprintf('--------------------------------------------------\n')
        
        % Print the table data
        rank_list = [rank_list,BLS_Rec(z,i)];
        fprintf('BLS      ')
        fprintf(2,formatSpec, BLS_Rec(z,i))
        fprintf(formatSpec, BLS_Pre(z,i))
        fprintf(2,formatSpec, BLS_F1(z,i))
        fprintf(formatSpec, BLS_Par(z,i))
        fprintf(formatSpec, BLS_time(z,i))
        fprintf('\n')

        if any(strcmpi('SVSA',compared_method))
            rank_list = [rank_list,BLSSVSA_Rec(z,i)];
            fprintf('SVSA     ')
            fprintf(2,formatSpec, BLSSVSA_Rec(z,i))
            fprintf(formatSpec, BLSSVSA_Pre(z,i))
            fprintf(2,formatSpec, BLSSVSA_F1(z,i))
            fprintf(formatSpec, BLSSVSA_Par(z,i))
            fprintf(formatSpec, BLSSVSA_time(z,i))
            fprintf('\n')
        end

        if any(strcmpi('GVSA',compared_method))
            rank_list = [rank_list,BLSGVSA_Rec(z,i)];
            fprintf('GVSA     ')
            fprintf(2,formatSpec, BLSGVSA_Rec(z,i))
            fprintf(formatSpec, BLSGVSA_Pre(z,i))
            fprintf(2,formatSpec, BLSGVSA_F1(z,i))
            fprintf(formatSpec, BLSGVSA_Par(z,i))
            fprintf(formatSpec, BLSGVSA_time(z,i))
            fprintf('\n')
        end

        if any(strcmpi('ESA',compared_method))
            rank_list = [rank_list,BLSESA_Rec(z,i)];
            fprintf('EET      ')
            fprintf(2,formatSpec, BLSESA_Rec(z,i))
            fprintf(formatSpec, BLSESA_Pre(z,i))
            fprintf(2,formatSpec, BLSESA_F1(z,i))
            fprintf(formatSpec, BLSESA_Par(z,i))
            fprintf(formatSpec, BLSESA_time(z,i))
            fprintf('\n')
        end

        if any(strcmpi('TSA',compared_method))
            rank_list = [rank_list,BLSTSA_Rec(z,i)];
            fprintf('TSA      ')
            fprintf(2,formatSpec, BLSTSA_Rec(z,i))
            fprintf(formatSpec, BLSTSA_Pre(z,i))
            fprintf(2,formatSpec, BLSTSA_F1(z,i))
            fprintf(formatSpec, BLSTSA_Par(z,i))
            fprintf(formatSpec, BLSTSA_time(z,i))
            fprintf('\n')
        end

        if any(strcmpi('FSA',compared_method))
            rank_list = [rank_list,BLSFSA_Rec(z,i)];
            fprintf('ours     ')
            fprintf(2,formatSpec, BLSFSA_Rec(z,i))
            fprintf(formatSpec, BLSFSA_Pre(z,i))
            fprintf(2,formatSpec, BLSFSA_F1(z,i))
            fprintf(formatSpec, BLSFSA_Par(z,i))
            fprintf(formatSpec, BLSFSA_time(z,i))
        fprintf('\n')
        end
        
        [~,index] = sort(rank_list,'descend');
        rank = find(index==length(rank_list));
        fprintf('--------------------------------------------------\n')
        fprintf(['Our methods is ',num2str(BLSFSA_Rec(z,i)),' and rank is ',num2str(rank), '\n'])
        fprintf('--------------------------------------------------\n')
    end
end

file = 'Results\TE\SACF\';
mkdir (file);
file_name = ['SACF_',num2str(Repet_time),'.mat'];
save([file,file_name], 'BLS_Pre',   'BLS_Rec',   'BLS_F1',   'BLS_Par',   'BLS_time',...
           'BLSTSA_Pre','BLSTSA_Rec','BLSTSA_F1','BLSTSA_Par','BLSTSA_time',...
           'BLSFSA_Pre','BLSFSA_Rec','BLSFSA_F1','BLSFSA_Par','BLSFSA_time',...
           'BLSGVSA_Pre','BLSGVSA_Rec','BLSGVSA_F1','BLSGVSA_Par','BLSGVSA_time',...
           'BLSSVSA_Pre','BLSSVSA_Rec','BLSSVSA_F1','BLSSVSA_Par','BLSSVSA_time',...
           'BLSESA_Pre','BLSESA_Rec','BLSESA_F1','BLSESA_Par','BLSESA_time')

%% plot recall, precision, maroc-F1 , time, parameter 
if ifplot
    nodes_line = 1:1:test_step;
    BLS_mean = [mean(BLS_Rec);mean(BLS_F1);mean(BLS_time);mean(BLS_Par)];
    BLSFSA_mean = [mean(BLSFSA_Rec);mean(BLSFSA_F1);mean(BLSFSA_time);mean(BLSFSA_Par)];
    BLSTSA_mean = [mean(BLSTSA_Rec);mean(BLSTSA_F1);mean(BLSTSA_time);mean(BLSTSA_Par)];  
    BLSGVSA_mean = [mean(BLSGVSA_Rec);mean(BLSGVSA_F1);mean(BLSGVSA_time);mean(BLSGVSA_Par)];   
    BLSSVSA_mean = [mean(BLSSVSA_Rec);mean(BLSSVSA_F1);mean(BLSSVSA_time);mean(BLSSVSA_Par)];   
    BLSESA_mean = [mean(BLSESA_Rec);mean(BLSESA_F1);mean(BLSESA_time);mean(BLSESA_Par)];    
    for i =1:4
        subplot(4,1,i);
        % BLS        
        plot(BLS_mean(i,:),'b')
        hold on;
    
        % BLS + FPD_SA 
        plot(BLSFSA_mean(i,:),'r')
        hold on;
        
        % BLS + traditional SA
        plot(BLSTSA_mean(i,:),'g')
        hold on;

        % BLS + traditional SA
        plot(BLSSVSA_mean(i,:),'k')
        hold on;

        % BLS + traditional SA
        plot(BLSGVSA_mean(i,:),'m')
        hold on;

        % BLS + traditional SA
        plot(BLSESA_mean(i,:),'c')
        hold on;
    end

end

disp('Finish the Demo!')