classdef SASO_Class_online
    properties
        Name = 'Increamental Broad Learning System';
        


        
        
        NumPerWin            % Feature nodes  per window
        NumWindow            % Number of windows of feature nodes
        NumEnhance           % Number of enhancement nodes
        NumAddFea         % Number of feature nodes per increment step 
        NumAddRel      % Number of enhancement nodes related to the incremental feature nodes per increment step
        NumAddEnh         % Number of enhancement nodes in each incremental learning
        IncStep              % Steps of incremental learning
        ShrScale             % The shrinkage scale of the enhancement nodes
        Scale
        RelScale
        L2Param              % The L2 regularization parameter
        SpaInpFeaWei         % Sparce weight matrix from input to feature layer
        NormFeaTot = struct  % Normlization setting of feature layer
        FeaEnhWei            % Weight matrix from feature layer to enhance layer
        Beta                 % Weight matrix from feature layer and enhance layer to output
        TotFeaSpa
        A_Matrix_Train
        AllFeaAddEnhWei
        AddFeaRelWei
        AddEnhScale
        A_Inverse
        sigfun
        FeaPD
        AllPD
        InitMed
        BanNodes
        AddNodeStep
        AddDataStep
        AddDataScale

    end
    %% Functions and algorithm
    methods    
        %% Functions and algorithm
        function Obj = SASO_Class_online(NumPerWin,NumWindow,NumEnhance,NumAddFea,NumAddRel,NumAddEnh,ShrScale,L2Param,BanIndex,AddNodeStep,AddDataStep,sigfun,InitMed)
            Obj.NumPerWin = NumPerWin;
            Obj.NumWindow = NumWindow;
            Obj.NumEnhance = NumEnhance;          
            Obj.NumAddFea = NumAddFea;
            Obj.NumAddRel = NumAddRel;
            Obj.NumAddEnh = NumAddEnh;   
            Obj.ShrScale = ShrScale;
            Obj.L2Param = L2Param;
            Obj.sigfun = sigfun;
            Obj.InitMed = InitMed;
            Obj.BanNodes = BanIndex;
            Obj.AddNodeStep = AddNodeStep;
            Obj.AddDataStep = AddDataStep;
            
            
 
             
        end
        %% Train        
        function Obj = Train(Obj, Input, Target)
            %% feature nodes
            Input = zscore(Input')';
            InputMat = [Input .1 * ones(size(Input,1),1)];  %加了一列偏置后的输入
            Obj.TotFeaSpa=zeros(size(Input,1),Obj.NumWindow * Obj.NumPerWin);
            for i = 1:Obj.NumWindow
                InpFeaWei = MyClassTools.IntialMed(size(Input,2)+1,Obj.NumPerWin,Obj.InitMed);
                FeaPerWin = mapminmax(InputMat * InpFeaWei);
                Obj.SpaInpFeaWei{i} = Obj.Sparse_bls(FeaPerWin,InputMat,1e-3,50)';
                Fea_Temp = InputMat * Obj.SpaInpFeaWei{i};
                [Fea_Temp,NormFea]  =  mapminmax(Fea_Temp',0,1); 
                NormTotTemp(i) = NormFea;
                Obj.TotFeaSpa(:,Obj.NumPerWin*(i-1)+1:Obj.NumPerWin*i) = Fea_Temp';                
            end
            Obj.NormFeaTot = NormTotTemp;
%             clear FeaPerWin InputMat input Fea_Temp;            
            %% enhancement nodes
            FeaMat = [Obj.TotFeaSpa .1 * ones(size(Obj.TotFeaSpa,1),1)];
            if Obj.NumPerWin * Obj.NumWindow >= Obj.NumEnhance   %正交化
                Obj.FeaEnhWei = orth(MyClassTools.IntialMed(Obj.NumWindow * Obj.NumPerWin + 1,Obj.NumEnhance,Obj.InitMed));
            else
                Obj.FeaEnhWei = orth(MyClassTools.IntialMed(Obj.NumWindow * Obj.NumPerWin + 1,Obj.NumEnhance,Obj.InitMed)')';

            end
            Enhance = FeaMat * Obj.FeaEnhWei;
            clear FeaMat;
            Obj.ShrScale = Obj.ShrScale / max(max(Enhance));
            if strcmp(Obj.sigfun,'logsig')
                Enhance = logsig(Enhance * Obj.ShrScale);
            else            
                Enhance = tansig(Enhance * Obj.ShrScale);
            end
            Obj.A_Matrix_Train = [Obj.TotFeaSpa Enhance];            
            clear Enhance;
            Obj.A_Inverse =  (Obj.A_Matrix_Train' * Obj.A_Matrix_Train + eye(size(Obj.A_Matrix_Train',1)) * (Obj.L2Param)) \ ( Obj.A_Matrix_Train');
            Obj.Beta =       Obj.A_Inverse * Target;           
        end     

        %% data incremtal process
         function Obj = DataIncBLS(Obj,AddInput,AllTarget)
            %新的输入矩阵
            AddInput = zscore(AddInput')';
            AddInputMat = [AddInput .1 * ones(size(AddInput,1),1)];  %加了一列偏置后的输入 
            %得到初始模型在添加样本后的各层表达
            AddInputOriFea_Tot = [];
            for i = 1:Obj.NumWindow+Obj.AddNodeStep
                AddInputFea_Temp = AddInputMat * Obj.SpaInpFeaWei{i};
                AddInputFea_Temp  =  mapminmax('apply',AddInputFea_Temp',Obj.NormFeaTot(i))';
                AddInputOriFea_Tot = [AddInputOriFea_Tot AddInputFea_Temp];
            end
            Obj.TotFeaSpa = [Obj.TotFeaSpa;AddInputOriFea_Tot];
            
            if Obj.AddNodeStep == 0
                AddFeaMat = [AddInputOriFea_Tot .1 * ones(size(AddInputOriFea_Tot,1),1)];
                AddEnhance = AddFeaMat * Obj.FeaEnhWei;
    %             Obj.AddDataScale(Obj.AddDataStep) = Obj.ShrScale / max(max(AddEnhance));
                if strcmp(Obj.sigfun,'logsig')
                    AddEnhance = logsig(AddEnhance * Obj.ShrScale);
                elseif strcmp(Obj.sigfun,'tansig')            
                    AddEnhance = tansig(AddEnhance * Obj.ShrScale);
                end
                A_Matrix_Add = [AddInputOriFea_Tot AddEnhance];
            else
                AddInputOriFea = AddInputOriFea_Tot(:,1:Obj.NumWindow*Obj.NumPerWin);
                AddInputOriFeaMat = [AddInputOriFea .1 * ones(size(AddInputOriFea_Tot,1),1)];
                AddInputOriEnh = AddInputOriFeaMat * Obj.FeaEnhWei;
                if strcmp(Obj.sigfun,'logsig')
                    AddInputOriEnh = logsig(AddInputOriEnh * Obj.ShrScale);
                elseif strcmp(Obj.sigfun,'tansig')            
                    AddInputOriEnh = tansig(AddInputOriEnh * Obj.ShrScale);
                end
                A_Matrix_Add = [AddInputOriFea AddInputOriEnh];
                for j = 1:Obj.AddNodeStep                    
                    AddInputAddFea = AddInputOriFea_Tot(:,Obj.NumWindow*Obj.NumPerWin+(j-1)*Obj.NumAddFea+1:Obj.NumWindow*Obj.NumPerWin+j*Obj.NumAddFea);
                    AddInputAddFeaMat = [AddInputAddFea .1 * ones(size(AddInputAddFea,1),1)];
                    AddInputRel = AddInputAddFeaMat * Obj.AddFeaRelWei{j};                      
                    if strcmp(Obj.sigfun,'logsig')
                        AddInputRel = logsig(AddInputRel * Obj.RelScale(j));
                    else            
                        AddInputRel = tansig(AddInputRel * Obj.RelScale(j));
                    end
                    AddInputOriFea = AddInputOriFea_Tot(:,1:Obj.NumWindow*Obj.NumPerWin+j*Obj.NumAddFea);
                    AddInputAllFeaMat = [AddInputOriFea .1 * ones(size(AddInputAddFea,1),1)];
                    AddInputAddEnh = AddInputAllFeaMat * Obj.AllFeaAddEnhWei{j};
                    if strcmp(Obj.sigfun,'logsig')
                        AddInputAddEnh = logsig(AddInputAddEnh * Obj.AddEnhScale(j));
                    else            
                        AddInputAddEnh = tansig(AddInputAddEnh * Obj.AddEnhScale(j));
                    end   
                    A_Matrix_Add = [A_Matrix_Add,AddInputAddFea,AddInputRel,AddInputAddEnh];
                end
                
            end
            AddA_Inverse = (A_Matrix_Add'  *  A_Matrix_Add+eye(size(A_Matrix_Add',1)) * (Obj.L2Param)) \ ( A_Matrix_Add' );
            Obj.A_Inverse = [Obj.A_Inverse AddA_Inverse];
            Obj.Beta = Obj.A_Inverse * AllTarget;
            Obj.A_Matrix_Train = [Obj.A_Matrix_Train;A_Matrix_Add];
         end

         function output = GetOutput(Obj,Data)
            Data = zscore(Data')';
            InpMat = [Data .1 * ones(size(Data,1),1)];
            AllFea = zeros(size(Data,1),Obj.NumWindow * Obj.NumPerWin);
            clear Data
            for i=1:Obj.NumWindow
                FeatTemp = InpMat * Obj.SpaInpFeaWei{i};
                FeatTemp  =  mapminmax('apply',FeatTemp',Obj.NormFeaTot(i))'; 
                AllFea(:,Obj.NumPerWin*(i-1)+1:Obj.NumPerWin*i)=FeatTemp;
            end
            clear FeatTemp;           
            if Obj.AddNodeStep == 0   
                FeaMat = [AllFea .1 * ones(size(AllFea,1),1)];
                if strcmp(Obj.sigfun,'logsig')
                    AllEnh = logsig(FeaMat * Obj.FeaEnhWei * Obj.ShrScale);
                else            
                    AllEnh = tansig(FeaMat * Obj.FeaEnhWei * Obj.ShrScale);
                end
                AMatrix = [AllFea AllEnh];
                clear FeaMat;
                clear AllEnh;
                output = AMatrix * Obj.Beta;  
            else
                OriFea = AllFea;
                OriFeaMat = [OriFea .1 * ones(size(AllFea,1),1)];
                if strcmp(Obj.sigfun,'logsig')
                    AMatrix = [OriFea logsig(OriFeaMat * Obj.FeaEnhWei * Obj.ShrScale)];
                else            
                    AMatrix = [OriFea tansig(OriFeaMat * Obj.FeaEnhWei * Obj.ShrScale)];
                end
                clear OriFeaMat AllFea
                for j=1:Obj.AddNodeStep
                    AddFea = InpMat * Obj.SpaInpFeaWei{Obj.NumWindow+j};
                    AddFea  =  mapminmax('apply',AddFea',Obj.NormFeaTot(Obj.NumWindow + j))';
                    if isempty(AddFea)
                        AddRel = [];
                    else
                        AddFeaMat = [AddFea .1 * ones(size(AddFea,1),1)];
                        if strcmp(Obj.sigfun,'logsig')
                            AddRel = logsig(AddFeaMat * Obj.AddFeaRelWei{j} * Obj.RelScale(j));
                        else            
                            AddRel = tansig(AddFeaMat * Obj.AddFeaRelWei{j} * Obj.RelScale(j));
                        end
                    end
                    OriFea = [OriFea AddFea];
                    FeaMat = [OriFea .1 * ones(size(OriFea,1),1)];
                    if strcmp(Obj.sigfun,'logsig')
                        AddEnh = logsig(FeaMat * Obj.AllFeaAddEnhWei{j} * Obj.AddEnhScale(j));
                    else            
                        AddEnh = tansig(FeaMat * Obj.AllFeaAddEnhWei{j} * Obj.AddEnhScale(j));
                    end                    
                    AddTot = [AddFea AddRel AddEnh];
                    AMatrix = [AMatrix AddTot];             
                end
                clear AddTot AllFea AddFea AddRel AddEnh FeaMat AddFeaMat InpMat
                output = AMatrix * Obj.Beta;
            end                         
        end   
                
        %% Nodes incremtal process
        function Obj = IncBLS(Obj,Input,Target)
            Input = zscore(Input')';
            InputMat = [Input .1 * ones(size(Input,1),1)];  %加了一列偏置后的输入            
            %随机添加输入到特征层的权值
            AddInpFeaWei = MyClassTools.IntialMed(size(Input,2)+1,Obj.NumAddFea,Obj.InitMed);

            AddFeature = mapminmax(InputMat * AddInpFeaWei);
            clear AddInpFeaWei Input;
            AddSapInpFeaWei  =  Obj.Sparse_bls(AddFeature,InputMat,1e-3,50)';
            Obj.SpaInpFeaWei{Obj.NumWindow + Obj.AddNodeStep} = AddSapInpFeaWei;
            clear AddFeature;
            %得到添加的特征的值
            AddFeaSpa = InputMat * AddSapInpFeaWei;
            clear InputMat
            [AddFeaSpa,AddNormFea]  =  mapminmax(AddFeaSpa',-1,1);
            AddFeaSpa = AddFeaSpa';
            %只在添加的特征值范围内定制了一个归一化尺度NormPerStep供测试使用
            Obj.NormFeaTot(Obj.NumWindow + Obj.AddNodeStep) = AddNormFea;
            clear AddNormFea
            %全体特征值
            Obj.TotFeaSpa = [Obj.TotFeaSpa AddFeaSpa];
            %加入偏置的全体特征矩阵
            FeaMat = [Obj.TotFeaSpa .1 * ones(size(Obj.TotFeaSpa,1),1)];
            %加入偏置的增加部分的特征矩阵
            if isempty(AddFeaSpa)
                AddFeaSpa = [];
                AddRel = [];
            else
                AddFeaMat = [AddFeaSpa .1 * ones(size(AddFeaSpa,1),1)];
                %增加特征在增强层对应点的权值
                if Obj.NumAddFea >= Obj.NumAddRel
                    Obj.AddFeaRelWei{Obj.AddNodeStep} = orth(MyClassTools.IntialMed(Obj.NumAddFea+1,Obj.NumAddRel,Obj.InitMed));

                else
                    Obj.AddFeaRelWei{Obj.AddNodeStep} = orth(MyClassTools.IntialMed(Obj.NumAddFea+1,Obj.NumAddRel,Obj.InitMed)')';

                end
                %相应增加的增强层
                AddRel = AddFeaMat * Obj.AddFeaRelWei{Obj.AddNodeStep};
                Obj.RelScale(Obj.AddNodeStep) = Obj.ShrScale / max(max(AddRel));
                if strcmp(Obj.sigfun,'logsig')
                    AddRel = logsig(AddRel * Obj.RelScale(Obj.AddNodeStep));
                else            
                    AddRel = tansig(AddRel * Obj.RelScale(Obj.AddNodeStep));
                end
                clear AddFeaMat;                
            end
            %全体特征到新增增强层的权值
            if Obj.NumWindow*Obj.NumPerWin+Obj.AddNodeStep*Obj.NumAddFea >= Obj.NumAddEnh
                Obj.AllFeaAddEnhWei{Obj.AddNodeStep} = orth(MyClassTools.IntialMed(Obj.NumWindow*Obj.NumPerWin+Obj.AddNodeStep*Obj.NumAddFea+1,Obj.NumAddEnh,Obj.InitMed));
            else
                Obj.AllFeaAddEnhWei{Obj.AddNodeStep} = orth(MyClassTools.IntialMed(Obj.NumWindow*Obj.NumPerWin+Obj.AddNodeStep*Obj.NumAddFea+1,Obj.NumAddEnh,Obj.InitMed)')';
            end
            %全体新增增强层的值，定制激活函数的缩放尺度Scale
            AddEnh = FeaMat * Obj.AllFeaAddEnhWei{Obj.AddNodeStep};
            Obj.AddEnhScale(Obj.AddNodeStep) = Obj.ShrScale / max(max(AddEnh));
            if strcmp(Obj.sigfun,'logsig')
                AddEnh = logsig(AddEnh * Obj.AddEnhScale(Obj.AddNodeStep));
            else            
                AddEnh = tansig(AddEnh * Obj.AddEnhScale(Obj.AddNodeStep));
            end
            clear FeaMat
            %全体A矩阵
            A_Matrix_Add = [AddFeaSpa AddRel AddEnh];
            A_Matrix_Tot = [Obj.A_Matrix_Train A_Matrix_Add];
            clear AddFeaSpa AddRel AddEnh
            Vec_D = Obj.A_Inverse * A_Matrix_Add;
            Vec_C = A_Matrix_Add - Obj.A_Matrix_Train * Vec_D;
            clear A_Matrix_Add
            if all(Vec_C(:)==0)
                [~,w] = size(Vec_D);
                Vec_B = (eye(w)-Vec_D'*Vec_D)\(Vec_D'*Obj.A_Inverse);
            else
                Vec_B = (Vec_C'  *  Vec_C+eye(size(Vec_C',1)) * (Obj.L2Param)) \ ( Vec_C' );
            end
            Obj.A_Inverse = [Obj.A_Inverse-Vec_D*Vec_B;Vec_B];
            clear Vec_B Vec_C Vec_D
            Obj.Beta = Obj.A_Inverse * Target; 
%             Obj.SpaInpFeaWei = [Obj.SpaInpFeaWei AddSapInpFeaWei];
            Obj.A_Matrix_Train = A_Matrix_Tot;                    
        end

         
        %% Test
        
        
       
        function [Obj,output] = PrunOutput(Obj,Data,BanType,Target,UType)
            Data = zscore(Data')';
            InpMat = [Data .1 * ones(size(Data,1),1)];
            OriFea=zeros(size(Data,1),Obj.NumWindow * Obj.NumPerWin);
            clear Data
            for i=1:Obj.NumWindow
                AddFea = InpMat * Obj.SpaInpFeaWei{i};                
                AddFea  =  mapminmax('apply',AddFea',Obj.NormFeaTot(i))'; 
                OriFea(:,Obj.NumPerWin*(i-1)+1:Obj.NumPerWin*i)=AddFea;
            end
            OriFeaMat = [OriFea .1 * ones(size(OriFea,1),1)];
            if strcmp(Obj.sigfun,'logsig')
                OriEnh = logsig(OriFeaMat * Obj.FeaEnhWei * Obj.ShrScale);
            else            
                OriEnh = tansig(OriFeaMat * Obj.FeaEnhWei * Obj.ShrScale);
            end            
            OriAMatrix = [OriFea OriEnh];   
            clear OriEnh
            OriFeaSA0 = OriFea;
            if strcmp(BanType,'FeatureNodes') || strcmp(BanType,'All')
                OriBanFea = Obj.BanNodes(Obj.BanNodes <= Obj.NumWindow * Obj.NumPerWin);
                OriFeaSA0(:,OriBanFea) = 0 ; 
            end
            
            clear OutputFeature_Temp;
            OriFeaMat0 = [OriFeaSA0 .1 * ones(size(OriFeaSA0,1),1)];
            if strcmp(Obj.sigfun,'logsig')
                OriEnhSA0 = logsig(OriFeaMat0 * Obj.FeaEnhWei * Obj.ShrScale);
            else            
                OriEnhSA0 = tansig(OriFeaMat0 * Obj.FeaEnhWei * Obj.ShrScale);
            end
            
            OriAMatrix0 = [OriFeaSA0 OriEnhSA0];            
            AllAddFeaSA0 = [];
            clear OriEnhSA0
            %% Incremental learning
            if Obj.AddNodeStep > 0
                for i = 1:Obj.AddNodeStep
                    %每一步增加的特征层
                    AddFea = InpMat * Obj.SpaInpFeaWei{i+Obj.NumWindow};                
                    AddFea  =  mapminmax('apply',AddFea',Obj.NormFeaTot(i+Obj.NumWindow))'; 
                    %每一步增加相应的增强层               
                    AddFeaMat = [AddFea .1 * ones(size(AddFea,1),1)];
                    if strcmp(Obj.sigfun,'logsig')
                        AddRel = logsig(AddFeaMat * Obj.AddFeaRelWei{i} * Obj.RelScale(i));
                    else            
                        AddRel = tansig(AddFeaMat * Obj.AddFeaRelWei{i} * Obj.RelScale(i));
                    end                    
                    %每一步增加的增强层
                    OriFea = [OriFea AddFea];
                    OriFeaMat = [OriFea .1 * ones(size(OriFea,1),1)];
                    if strcmp(Obj.sigfun,'logsig')
                        AddEnh = logsig(OriFeaMat * Obj.AllFeaAddEnhWei{i} * Obj.AddEnhScale(i));
                    else            
                        AddEnh = tansig(OriFeaMat * Obj.AllFeaAddEnhWei{i} * Obj.AddEnhScale(i));
                    end
                    
                    AMatrix = [OriAMatrix AddFea AddRel AddEnh];
                    AddFeaSA0 = AddFea;
                    AllAddFeaSA0 = [AllAddFeaSA0 AddFeaSA0];
                    clear  AddRel AddEnh AddFea OriFeaMat 
                    if strcmp(BanType,'FeatureNodes') || strcmp(BanType,'All')                        
                        AddBanFea = Obj.BanNodes(Obj.BanNodes > Obj.NumWindow * Obj.NumPerWin + Obj.NumEnhance+(i-1) * (Obj.NumAddFea+Obj.NumAddEnh+Obj.NumAddRel)&...
                                                      Obj.BanNodes <= Obj.NumWindow * Obj.NumPerWin + Obj.NumEnhance+(i-1) * (Obj.NumAddFea+Obj.NumAddEnh+Obj.NumAddRel)+Obj.NumAddFea);
                        AddFeaSA0(:,AddBanFea-Obj.NumWindow * Obj.NumPerWin - Obj.NumEnhance-(i-1)*(Obj.NumAddFea+Obj.NumAddEnh+Obj.NumAddRel)) = 0;
                        clear AddBanFea
                        AddFeaMatSA0 = [AddFeaSA0 .1 * ones(size(AddFeaSA0,1),1)];
                        AllFeaMatSA0 = [OriFeaSA0 AllAddFeaSA0 .1 * ones(size(OriFeaSA0,1),1)];
                        if strcmp(Obj.sigfun,'logsig')
                            AddRelSA0 = logsig(AddFeaMatSA0 * Obj.AddFeaRelWei{i} * Obj.RelScale(i));
                            AddEnhSA0 = logsig(AllFeaMatSA0 * Obj.AllFeaAddEnhWei{i} * Obj.AddEnhScale(i));
                        else            
                            AddRelSA0 = tansig(AddFeaMatSA0 * Obj.AddFeaRelWei{i} * Obj.RelScale(i));
                            AddEnhSA0 = tansig(AllFeaMatSA0 * Obj.AllFeaAddEnhWei{i} * Obj.AddEnhScale(i));
                        end  
                        clear AddFeaMatSA0 AllFeaMatSA0
                        OriAMatrix0 = [OriAMatrix0 AddFeaSA0 AddRelSA0 AddEnhSA0];
                    end 
                end 
            end
            clear AddFeaSA0 AddRelSA0 AddEnhSA0 AllFeaMatSA0 AddFeaMatSA0 AddBanFea OriAMatrix
            if strcmp(BanType,'EnhanNodes')
                AMatrix(:,Obj.BanNodes) = [] ; 
            else
                OriAMatrix0(:, Obj.BanNodes) = [];
                AMatrix = OriAMatrix0;
            end   
            clear OriAMatrix0
            if strcmp(UType,'update')
                Obj.Beta = (AMatrix'  *  AMatrix+eye(size(AMatrix',1)) * (Obj.L2Param)) \ ( AMatrix'  *  Target);
%                 AR_ = Obj.A_Inverse;
%                 AR_(Obj.BanNodes,:) = [] ;
%                 AD = Obj.A_Matrix_Train(:,Obj.BanNodes);
%                 BT = Obj.A_Inverse(Obj.BanNodes,:);
%                 Obj.Beta = AR_ * (eye(size(AD,1)) - AD * BT \ Target) ;
            end              
                
            clear FeaMat;
            clear OutputEnhance;
            output = AMatrix * Obj.Beta; 
        end
               
        function wk = Sparse_bls(Obj,A,b,lam,itrs)
            AA = (A') * A;
            m = size(A,2);
            n = size(b,2);
            x = zeros(m,n);
            wk = x; 
            ok=x;
            uk=x;
            L1=eye(m)/(AA+eye(m));
            L2=L1*A'*b; 
            for i = 1:itrs
                tempc=ok-uk;
                ck =  L2+L1*tempc;
                ok = max( ck+uk - lam,0 ) - max( -ck-uk - lam ,0);
                uk=uk+(ck-ok);
                wk=ok;
            end
        end 


    end %method
end %class
         
        
