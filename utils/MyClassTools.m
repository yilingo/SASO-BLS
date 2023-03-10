%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fast Sensitivity Analysis Based Online Self-Organizing Broad Learning System (tools)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2022

classdef MyClassTools
    properties
        Name = 'tools';
    end
    methods (Static = true)

        %% class result
        function  y = ClassResult(x)
            for i=1:size(x,1)
            [~,y(i)]=max(x(i,:));
            end
            y=y';     
        end

        %% Normlization [-1 , 1]
        function [minn, maxx, X] = NormDm(X)
            % peforms linear normalization

            sizeX=size(X);
            minn=zeros(1, size(X,2));
            maxx=zeros(1, size(X,2));
            for i=1:sizeX(2)
                minn(i)=min(X(:,i));
                maxx(i)=max(X(:,i));
            end
            for ii=1:sizeX(1)
                for j=1:sizeX(2)
                    X(ii,j)=(((X(ii,j)-minn(j))/(maxx(j)-minn(j))))*2-1;
                end
            end
        end

        %% Normlization [0 , 1]
        function [minn, maxx, X] = NormDp(X)
            % peforms linear normalization

            sizeX=size(X);
            minn=zeros(1, size(X,2));
            maxx=zeros(1, size(X,2));
            for i=1:sizeX(2)
                minn(i)=min(X(:,i));
                maxx(i)=max(X(:,i));
            end
            for ii=1:sizeX(1)                    
                for j=1:sizeX(2) 
                    if maxx(j)-minn(j) == 0
                        X(ii,j)=0;
                    else
                        X(ii,j)=((X(ii,j)-minn(j))/(maxx(j)-minn(j)));
                    end
                end
            end
        end
        
         %% Normlization adapt [-1 , 1]
        function X=normDadaptm(X, minn, maxx)
            sizeX=size(X);
            for ii=1:sizeX(1)
                for j=1:sizeX(2)
                    X(ii,j)=(((X(ii,j)-minn(j))/(maxx(j)-minn(j))))*2-1;
                end
            end
        end

        %% Normlization adapt [0 , 1]
        function X=normDadaptp(X, minn, maxx)
        sizeX=size(X);
        for ii=1:sizeX(1)
            for j=1:sizeX(2)
                if maxx(j)-minn(j) ==0
                    X(ii,j) = 0;
                else
                    X(ii,j)=(((X(ii,j)-minn(j))/(maxx(j)-minn(j))));
                end
            end
        end
        end
        
        %% change label
        function Changed_label = ChangeLabel(Y)
            Changed_label = zeros(length(Y),max(Y));
            for i = 1:length(Y)
                for j = 1:length(max(Y))
                    Changed_label(i,Y(i)) = 1;
                end
            end
        end

        %% sample select for SA
        function SampleSelInd = SampleSel(TrainY,LabelY,NumSel)
            ClassSeq = 1:max(TrainY);
            SampleSelInd = [];
            for i = ClassSeq
                SeqTrainInd{i} = find(ismember(TrainY, ClassSeq(i)));
                SeqLabelInd{i} = find(ismember(LabelY, ClassSeq(i)));
                ConformInd = intersect(SeqTrainInd{i},SeqLabelInd{i});
                try
                    random_num = ConformInd(randperm(numel(ConformInd),NumSel));
                catch
                    random_num = SeqTrainInd{i}(randperm(numel(SeqTrainInd{i}),NumSel));
                end
                SampleSelInd = [SampleSelInd; random_num];
            end            
        end
        

        %% initial method select
        function Wei = IntialMed(InpDim,OutDim,Med)
            if strcmp(Med,'MeanX')
                Wei = unifrnd(-sqrt(6/(InpDim+OutDim)),sqrt(6/(InpDim+OutDim)),InpDim, OutDim);
            elseif strcmp(Med,'GuassX')
                Wei = normrnd(0,sqrt(2/(InpDim+OutDim)),[InpDim,OutDim]); 
            elseif strcmp(Med,'MeanHe')
                Wei = unifrnd(-sqrt(3/2*(InpDim+OutDim)),sqrt(3/2*(InpDim+OutDim)),InpDim, OutDim);
            elseif strcmp(Med,'GuassHe')
                Wei = normrnd(0,sqrt(2/(InpDim)),[InpDim,OutDim]);

            end
        end

        %% parameter calculate
        function num_para = bls_parameters(model,Med,State)
            data_dim = length(model.SpaInpFeaWei{1}(:,1));

            if strcmp(State,'offline')
                Step = model.Step;
            elseif strcmp(State,'online')
                Step = model.AddNodeStep;
            else
                error('Input wrong state!');
            end

            if strcmp(Med,'bls')   
                if Step == 0 
                    num_para = data_dim*(model.NumPerWin*model.NumWindow+model.NumAddFea*Step)+...
                    ((model.NumPerWin+1)*model.NumWindow*model.NumAddEnh) + length(model.Beta(1,:))*length(model.Beta(:,1));
                else
                    num_para = data_dim*(model.NumPerWin*model.NumWindow+model.NumAddFea*Step)+... % input to feature
                    ((model.NumPerWin+1)*model.NumWindow*model.NumEnhance)+...    % ori feature to ori enhance
                    (Step*model.NumPerWin*model.NumWindow+(1+Step) * Step/2*model.NumAddFea+Step)*model.NumAddEnh+... % all feature to added enhance
                    (model.NumAddFea+1)*model.NumAddRel*Step + length(model.Beta(1,:))*length(model.Beta(:,1)); % added feature to related enhance
                end
            elseif strcmp(Med,'saso-bls')
                if Step == 0                        
                    NumBanPer = hist(model.BanNodes,linspace(model.NumPerWin/2,model.NumPerWin*3/2,2));
                    num_para = data_dim*(model.NumPerWin*model.NumWindow-NumBanPer(1)+model.NumAddFea*Step)+...
                    (((model.NumPerWin+1)*model.NumWindow-NumBanPer(1))*(model.NumAddEnh-NumBanPer(2))) +...
                    length(model.Beta(1,:))*(length(model.Beta(:,1))-sum(NumBanPer));
                else
                    NumBanPer = hist(model.BanNodes,linspace(model.NumPerWin/2,model.NumPerWin/2*(5+Step*4),2+3*Step));
                    % all feature to added enhance
                    para_allf_addenhance = 0;
                    para_addf_addrel = 0;
                    for i = 1:Step
                        para_allf_addenhance = para_allf_addenhance + (model.NumPerWin*model.NumWindow-NumBanPer(1) + model.NumAddFea - NumBanPer(end-3*i)+1)...
                            * (model.NumAddEnh-NumBanPer(end-3*(i-1)-2));
                        para_addf_addrel = para_addf_addrel + (model.NumAddFea- NumBanPer(end-3*(i-1)-2) +1)*(model.NumAddRel- NumBanPer(end-3*(i-1)-1));
                    end
                    num_para = data_dim*(model.NumPerWin*model.NumWindow+model.NumAddFea*Step-sum(NumBanPer(1:2)))+...
                    (((model.NumPerWin+1)*model.NumWindow-NumBanPer(1))*(model.NumEnhance-NumBanPer(2)))+...  
                    para_allf_addenhance + para_addf_addrel + length(model.Beta(1,:))*(length(model.Beta(:,1))-sum(NumBanPer));

                end
            end

            num_para = num_para/1000;
        end           
        
    end % method
end % class