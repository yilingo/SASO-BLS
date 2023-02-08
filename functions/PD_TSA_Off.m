classdef PD_TSA_Off
    properties
        Name = 'Off-line Broad Learning System';

    end
    %% Functions and algorithm
    methods    (Static = true)              
        function model = TSA(model,SelTrainA,sigfun)
            NumFeaOri = model.NumPerWin * model.NumWindow;
            NumEnhOri = model.NumEnhance; 
            EnhMatSelOri = SelTrainA(:,NumFeaOri + 1:NumFeaOri+NumEnhOri);
            NumClass = length(model.Beta(1,:));
            if model.Step == 0      
                AddFeaPDD = model.Beta(1:NumFeaOri,:);
                AddEnhPDD = model.Beta(NumFeaOri + 1:NumFeaOri+NumEnhOri,:); 
                AddEnhSel = EnhMatSelOri;
                AddRelSel = [];
                AddFeaEnhWei = model.FeaEnhWei(1:NumFeaOri,:);
                OriFeaAddEnhWei = [];
                AddFeaRelWei = [];                
                AddRelPDD = [];     

                AddFeaEnhPDID3 = zeros(NumFeaOri,NumClass,length(SelTrainA(:,1)));
                AddFeaRelPDID3 = [];

            else
                NumAddFea = model.NumAddFea;
                NumAddRel = model.NumAddRel;
                NumAddEnh = model.NumAddEnh;
                AddFeaPDD = model.Beta(end - (NumAddFea+NumAddEnh+NumAddRel) + 1:end - (NumAddEnh+NumAddRel),:);
                AddEnhSel = SelTrainA(:,end-NumAddEnh+1:end); 
                AddRelSel = SelTrainA(:,end-NumAddRel-NumAddEnh+1:end-NumAddEnh);  
                AddFeaEnhWei = model.AllFeaAddEnhWei{model.Step}(end-NumAddFea:end-1,:);
                OriFeaAddEnhWei = model.AllFeaAddEnhWei{model.Step}(1:end-NumAddFea-1,:);
                AddFeaRelWei = model.AddFeaRelWei{model.Step}(1:NumAddFea,:);                
                AddRelPDD = model.Beta(end - NumAddRel - NumAddEnh+1:end -NumAddEnh,:);
                AddEnhPDD = model.Beta(end-NumAddEnh+1:end,:); 

                AddFeaEnhPDID3 = zeros(NumAddEnh,NumClass,length(SelTrainA(:,1)));
                AddFeaRelPDID3 = zeros(NumAddRel,NumClass,length(SelTrainA(:,1)));
            end 
            for z = 1:length(SelTrainA(:,1))   
                if strcmp(sigfun,'logsig')
                    DiagAddEnh = diag(AddEnhSel(z,:) .*(1-AddEnhSel(z,:)));                    
                    try 
                        DiagAddRel = diag(AddRelSel(z,:) .*(1-AddRelSel(z,:)));
                    catch
                        DiagAddRel = [];
                    end
                else
                    DiagAddEnh = diag(1-AddEnhSel(z,:) .* AddEnhSel(z,:));                       
                    try
                        DiagAddRel = diag(1-AddRelSel(z,:) .* AddRelSel(z,:));                           
                    catch
                        DiagAddRel = [];
                    end
                end
                if model.Step == 0            
                      AddFeaEnhPDID3(:,:,z) = AddFeaEnhWei * DiagAddEnh * AddEnhPDD;
                else
                    AddFeaEnhPDID3(:,:,z) = AddFeaEnhWei * DiagAddEnh * AddEnhPDD;
                    AddFeaRelPDID3(:,:,z) = AddFeaRelWei * DiagAddRel * AddRelPDD;
                end
            end


            if model.Step == 0 
                FeaEnhSMatrixInDSel= sqrt(sum(AddFeaEnhPDID3.^2,3)./length(SelTrainA(:,1)));
            else
                FeaEnhSMatrixInDSel = sqrt(sum(AddFeaEnhPDID3.^2,3)./length(SelTrainA(:,1)));
                RelEnhanSMatrixInDSel = sqrt(sum(AddFeaRelPDID3.^2,3)./length(SelTrainA(:,1)));
            end

            if model.Step == 0 
                model.FeaPD = FeaEnhSMatrixInDSel + AddFeaPDD;
                model.AllPD = [model.FeaPD;AddEnhPDD];
            else
                OriFeaPD = model.FeaPD + OriFeaAddEnhWei * DiagAddEnh * AddEnhPDD; 
                AddFeaPD = AddFeaPDD + FeaEnhSMatrixInDSel + RelEnhanSMatrixInDSel;
                model.FeaPD = [OriFeaPD;AddFeaPD];
                for k = 1:model.Step-1
                        model.AllPD(NumFeaOri+NumEnhOri+(NumAddFea+NumAddRel+NumAddEnh)*(k-1)+1:NumFeaOri+NumEnhOri+(NumAddFea+NumAddRel+NumAddEnh)*(k-1)+NumAddFea,:)...
                      = model.FeaPD(NumFeaOri+NumAddFea*(k-1)+1:NumFeaOri+NumAddFea*k,:);
                end                    
                model.AllPD = [model.AllPD ; AddFeaPD ; AddRelPDD ; AddEnhPDD];
            end   
            FyFeature = max(model.AllPD,[],2);
            [FyFeature_sort, Sort_index] = sort(FyFeature,'descend');
            DeltaFeatureFy = zeros(length(FyFeature_sort)-1,1);
            for i = 1:(length(FyFeature_sort)-1)                
                DeltaFeatureFy(i) = FyFeature_sort(i)-FyFeature_sort(i+1);
            end
            MeanFeatureFy = mean(DeltaFeatureFy);
            
            Condition1 = DeltaFeatureFy./FyFeature_sort(1:end-1);
            Condition1_judge = (Condition1>0.001);
            SelectIndex1 = find(Condition1_judge==1);
            SelectNeuron1 = Sort_index(SelectIndex1);

            Condition2_judge = (FyFeature_sort>1*MeanFeatureFy) ;
            SelectIndex2 = find(Condition2_judge==1);
            SelectNeuron2 = Sort_index(SelectIndex2);

            SelectNeruonSet = intersect(SelectNeuron1,SelectNeuron2);          
            model.BanNodes = setdiff(Sort_index,SelectNeruonSet);                      
        end
    end %methodq
end %class
         
        

         
        
