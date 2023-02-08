%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% SASO-BLS online mode  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%% DataSet Load 
DataSet = 'TE'; 
[ForTrain_x,ForTest_x,ForTrain_y,ForTest_y] = DataSetLoad.load(DataSet);

%% data process
IFOnline = true;
IFshuffle = false; 
ifplot = true;
NormMethod = 4;
OnlineIntrate = 1/8;
OnlineStep = 1/OnlineIntrate-1;
[ForTrain_x,ForTest_x] = DataSetLoad.Norm(ForTrain_x,ForTest_x,NormMethod);

if IFOnline && ~IFshuffle
    [IniForTrain_x,IniForTrain_y,RestForTrain_x,RestForTrain_y] = DataSetLoad.OnlineSeg(ForTrain_x,ForTrain_y,OnlineIntrate,OnlineStep);
elseif IFOnline && IFshuffle
    [IniForTrain_x,IniForTrain_y,RestForTrain_x,RestForTrain_y] = DataSetLoad.OnlineSeg(ForTrain_x,ForTrain_y,OnlineIntrate,OnlineStep);
    [IniForTrain_x,IniForTrain_y] = DataSetLoad.Shuffle(IniForTrain_x,IniForTrain_y);
    [RestForTrain_x,RestForTrain_y] = DataSetLoad.Shuffle(RestForTrain_x,RestForTrain_y);
elseif ~IFOnline && IFshuffle
    [ForTrain_x,ForTrain_y] = DataSetLoad.Shuffle(ForTrain_x,ForTrain_y);
end


%% Parameter set 
NumEnhance =10; % Nodes number of the enhancement layer 
NumPerWin = 10;  %Nodes number of the feature mapping layer per window
NumWindow = 1;  % Number of windows of the feature mapping layer

% incremental learning
NumFeaPerInc = 10;
NumEnhRelPerInc = 10;
NumEnhPerInc = 10;

% other parameters
sigfun = 'tansig';
L2Param = 2^-30; %L2 parameter
ShrScale = .8;   %the l2 regularization parameter and the shrinkage scale of the enhancement nodes   
BanType = 'All'; %FeatureNodes %All %EnhanNodes
NumEech4SA = 1; %Sample number of each class for FPD-SA
AddNodeStep = 0;
AddDataStep = -1;
AllStep = 0;
BanIndex = [];
InitMed = 'GuassX'; %MeanX , GuassX, MeanHe,GuassHe



%% Model Initialization
Model = SASO_Class_online(NumPerWin,NumWindow,NumEnhance,NumFeaPerInc,NumEnhRelPerInc,NumEnhPerInc,ShrScale,L2Param,BanIndex,AddNodeStep,AddDataStep,sigfun,InitMed);

tic;
Model = Model.Train(IniForTrain_x,IniForTrain_y);
BLS_time(1,1) = toc;
NumAddData = length(IniForTrain_y(:,1));


%% Result sequence initalization
disp(['Parameters: ',InitMed,':***OnlineIntrate:',num2str(OnlineIntrate),'***OnlineStep:',num2str(OnlineStep)]);
while Model.AddDataStep <= OnlineStep-1  
    AllStep = AllStep + 1;
    Model.AddDataStep = Model.AddDataStep +1;    
    disp(['********Start the ', num2str(Model.AddDataStep), '-th learning process********']);
    if Model.AddDataStep ~= 0        
        AddTrain_x = RestForTrain_x{Model.AddDataStep};
        AddTrain_y = RestForTrain_y{Model.AddDataStep};
        IniForTrain_x = [IniForTrain_x;AddTrain_x];
        IniForTrain_y = [IniForTrain_y;AddTrain_y];
        tic;
        Model = Model.DataIncBLS(AddTrain_x,IniForTrain_y); 
        BLS_time(1,AllStep) = toc;        
        NumAddData = length(AddTrain_x(:,1));
    end 

    % Validation of BLS
    ValResult = Model.GetOutput(ForTest_x);
    ValResultDis = MyClassTools.ClassResult(ValResult);
    ValLabelDis = MyClassTools.ClassResult(ForTest_y);
    ValIndex = Evaluation_idx(ValResultDis,ValLabelDis);
    ValACC = ValIndex.Micro(); 
    [~,~,~,~,~,~,WMacro_P,WMacro_R,WMacro_F1] = ValIndex.Macro();
    BLSpara = MyClassTools.bls_parameters(Model,'bls','online');

    BLS_Pre(1,AllStep) = WMacro_P;
    BLS_Rec(1,AllStep) = WMacro_R;
    BLS_F1(1,AllStep) = WMacro_F1;
    BLS_Par(1,AllStep) = BLSpara;
    disp(['(BLS) Data incremental step:',num2str(Model.AddDataStep),'-----Nodes incremental step:',num2str(Model.AddNodeStep)]);
    disp(['The Precision of BLS is ' ,num2str(WMacro_P)]);
    fprintf(2,['The Recall of BLS is ' ,num2str(WMacro_R),'\n']);
    fprintf(2,['The macro-F1 of BLS is ' ,num2str(WMacro_F1),'\n']);
    disp(['The parameter of BLS is ' ,num2str(BLSpara),'K']);

    % selected train all matrix
    TrainLabelDis = MyClassTools.ClassResult(IniForTrain_y);
    NumEachLabel = tabulate(TrainLabelDis(end - NumAddData+1:end));
    NumEech4SA = min(NumEachLabel(:,2));
    SelTrainA = Model.A_Matrix_Train;

    % FPD-SA for compression
    mode = 'AD';
    tic;
    Model = FPD_SA_Online.SA(Model,SelTrainA,NumEech4SA,sigfun,mode); 
    SASOBLS_time(1,AllStep) = toc + BLS_time(1,AllStep);


    [ModelFSA,FSATrainResult] = Model.PrunOutput(IniForTrain_x,BanType,IniForTrain_y,'update');
    FSATrainResultDis = MyClassTools.ClassResult(FSATrainResult);
    FSATrainIndex = Evaluation_idx(FSATrainResultDis,TrainLabelDis);
    FSATrainACC = FSATrainIndex.Micro(); 
    SASOBLS_train_acc(1,AllStep) = FSATrainACC*length(TrainLabelDis)/length(ForTrain_x);

    [~,FSAValResult] = ModelFSA.PrunOutput(ForTest_x,BanType,ForTest_y,'test');
    FSAValResultDis = MyClassTools.ClassResult(FSAValResult);
    FSAValIndex = Evaluation_idx(FSAValResultDis,ValLabelDis);
    [~,~,~,~,~,~,WMacro_P,WMacro_R,WMacro_F1]  = FSAValIndex.Macro();
    FSABLSpara = MyClassTools.bls_parameters(Model,'saso-bls','online');

    % FPD-SA output and save
    BLSFSA_Pre(1,AllStep) = WMacro_P;
    BLSFSA_Rec(1,AllStep) = WMacro_R;
    BLSFSA_F1(1,AllStep) = WMacro_F1;
    BLSFSA_Par(1,AllStep) = FSABLSpara;
    disp(['(SASO-BLS) Data incremental step:',num2str(Model.AddDataStep),'-----Nodes incremental step:',num2str(Model.AddNodeStep)]);
    disp(['The Precision of BLS is ' ,num2str(WMacro_P)]);
    fprintf(2,['The Recall of SASO-BLS is -------' ,num2str(WMacro_R),'\n']);
    fprintf(2,['The macro-F1 of SASO-BLS is ' ,num2str(WMacro_F1),'\n']);
    disp(['The parameter of SASO-BLS is -------' ,num2str(FSABLSpara),'K']);
    
    if BLSFSA_Rec(AllStep) == max(BLSFSA_Rec)
            ModelBest = ModelFSA;
    end

    % Nodes incremental process
    AddNodeStepThis = 0;
    while Model.AddDataStep <= OnlineStep
        ModelBefore = ModelFSA;
        Model.AddNodeStep = Model.AddNodeStep +1;  
        AddNodeStepThis = AddNodeStepThis + 1;
        AllStep = AllStep + 1;
        tic;
        Model = Model.IncBLS(IniForTrain_x,IniForTrain_y);
        BLS_time(1,AllStep) = toc;

        TrainLabelDis = MyClassTools.ClassResult(IniForTrain_y);


        % BLS Validation
        ValResult = Model.GetOutput(ForTest_x);
        ValResultDis = MyClassTools.ClassResult(ValResult);
        ValLabelDis = MyClassTools.ClassResult(ForTest_y);
        ValIndex = Evaluation_idx(ValResultDis,ValLabelDis);
        [~,~,~,~,~,~,WMacro_P,WMacro_R,WMacro_F1] = ValIndex.Macro();
        BLSpara = MyClassTools.bls_parameters(Model,'bls','online');
        
        BLS_Pre(1,AllStep) = WMacro_P;
        BLS_Rec(1,AllStep) = WMacro_R;
        BLS_F1(1,AllStep) = WMacro_F1;
        BLS_Par(1,AllStep) = BLSpara;
        disp(['(BLS) Data incremental step:',num2str(Model.AddDataStep),'-----Nodes incremental step:',num2str(Model.AddNodeStep)]);
        disp(['The Precision of BLS is ' ,num2str(WMacro_P)]);
        fprintf(2,['The Recall of BLS is ' ,num2str(WMacro_R),'\n']);
        fprintf(2,['The macro-F1 of BLS is ' ,num2str(WMacro_F1),'\n']);
        disp(['The parameter of BLS is ' ,num2str(BLSpara),'K']);
        
        mode = 'AN';
        SelTrainA = Model.A_Matrix_Train;
        tic;
        Model = FPD_SA_Online.SA(Model,SelTrainA,NumEech4SA,sigfun,mode); 
        SASOBLS_time(1,AllStep) = toc+BLS_time(1,AllStep);


        [ModelFSA,FSATrainResult] = Model.PrunOutput(IniForTrain_x,BanType,IniForTrain_y,'update');
        FSATrainResultDis = MyClassTools.ClassResult(FSATrainResult);
        FSATrainIndex = Evaluation_idx(FSATrainResultDis,TrainLabelDis);
        FSATrainACC = FSATrainIndex.Micro(); 
        SASOBLS_train_acc(1,AllStep) = FSATrainACC*length(TrainLabelDis)/length(ForTrain_x);

        [~,FSAValResult] = ModelFSA.PrunOutput(ForTest_x,BanType,ForTest_y,'test');
        FSAValResultDis = MyClassTools.ClassResult(FSAValResult);
        FSAValIndex = Evaluation_idx(FSAValResultDis,ValLabelDis);
        [~,~,~,~,~,~,WMacro_P,WMacro_R,WMacro_F1]  = FSAValIndex.Macro();
        FSABLSpara = MyClassTools.bls_parameters(Model,'saso-bls','online');

        % FPD-SA output and save
        BLSFSA_Pre(1,AllStep) = WMacro_P;
        BLSFSA_Rec(1,AllStep) = WMacro_R;
        BLSFSA_F1(1,AllStep) = WMacro_F1;
        BLSFSA_Par(1,AllStep) = FSABLSpara;
        disp(['(SASO-BLS) Data incremental step:',num2str(Model.AddDataStep),'-----Nodes incremental step:',num2str(Model.AddNodeStep)]);
        disp(['The Precision of SASO-BLS is -------' ,num2str(WMacro_P)]);
        fprintf(2,['The Recall of SASO-BLS is -------' ,num2str(WMacro_R),'\n']);
        fprintf(2,['The macro-F1 of SASO-BLS is ' ,num2str(WMacro_F1),'\n']);
        disp(['The parameter of FSA SASO-BLS -------' ,num2str(FSABLSpara),'K']);


        one_order = diff(SASOBLS_train_acc);
        if BLSFSA_Rec(end) == max(BLSFSA_Rec)
            ModelBest = ModelFSA;
        end
        if AllStep > 2 && (one_order(end) < 0.005 )
            if Model.AddDataStep == OnlineStep && AddNodeStepThis < 2
                continue
            elseif AddNodeStepThis == 2
                break
            else
                Model = ModelBefore;
                break
            end
        end
    end
end

% Validation of BLS
[~,ValResult] = ModelBest.PrunOutput(ForTest_x,BanType,ForTest_y,'test');
ValResultDis = MyClassTools.ClassResult(ValResult);
ValLabelDis = MyClassTools.ClassResult(ForTest_y);
ValIndex = Evaluation_idx(ValResultDis,ValLabelDis);
ValACC = ValIndex.Micro(); 
figure;
plotconfusion(ForTest_y',ValResult');

if ifplot
    figure;
    BLS_indicator_list = [BLS_Rec;BLS_Pre;BLS_F1;BLS_time;BLS_Par];
    BLSFSA_indicator_list = [BLSFSA_Rec;BLSFSA_Pre;BLSFSA_F1;SASOBLS_time;BLSFSA_Par];
    for i =1:length(BLS_indicator_list(:,1))
        subplot(length(BLS_indicator_list(:,1)),1,i);
        % BLS        
        plot(BLS_indicator_list(i,:),'b')
        hold on;

        % SASO-BLS 
        plot(BLSFSA_indicator_list(i,:),'r')
    end

end
disp('Finish the Demo!')



