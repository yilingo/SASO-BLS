classdef Evaluation_idx
    properties
        Len_data
        con_mat
        order
    end
    methods 
        %% Functions and algorithm
        function Obj = Evaluation_idx(Output_vector,Label_vector)
            Obj.Len_data = length(Label_vector);
            [Obj.con_mat,Obj.order] = confusionmat(Label_vector,Output_vector);
        end
        
        
        %% micro
        function Micro_F1 = Micro(Obj)
            Micro_F1 = trace(Obj.con_mat)/Obj.Len_data;     
        end
        
        %% macro
        function [Macro_P,Macro_R,Macro_F1,Macro_PL,Macro_RL,Macro_F1L,WMacro_P,WMacro_R,WMacro_F1] = Macro(Obj)
            Macro_PL = zeros(1,length(Obj.order));
            Macro_RL = zeros(1,length(Obj.order));
            Macro_F1L = zeros(1,length(Obj.order));
            for i = 1:length(Obj.order)
                if sum(Obj.con_mat(:,i))~= 0 
                    Macro_PL(i) = Obj.con_mat(i,i)/sum(Obj.con_mat(:,i));
                else
                    Macro_PL(i) = 0;
                end
                if sum(Obj.con_mat(i,:))~= 0
                    Macro_RL(i) = Obj.con_mat(i,i)/sum(Obj.con_mat(i,:));                  
                else
                    Macro_RL(i) = 0 ;
                end  
                if Macro_PL(i)~=0 || Macro_RL(i)~=0
                    Macro_F1L(i) = 2.*Macro_PL(i).*Macro_RL(i)/(Macro_PL(i)+Macro_RL(i));
                else
                    Macro_F1L(i) = 0;
                end

            end
            Macro_P = sum(Macro_PL)/length(Obj.order);
            Macro_R = sum(Macro_RL)/length(Obj.order);
            Macro_F1 = sum(Macro_F1L)/length(Obj.order);  
            
            Num_EachClaL = sum(Obj.con_mat.');
            WMacro_P = sum(Num_EachClaL.*Macro_PL)/Obj.Len_data;
            WMacro_R = sum(Num_EachClaL.*Macro_RL)/Obj.Len_data;
            WMacro_F1 = sum(Num_EachClaL.*Macro_F1L)/Obj.Len_data;
        end
        
        function K_index = Kappa(Obj)
            P_O = Obj.Micro();
            P_ab = zeros(1,length(Obj.order));
            for k = 1:length(Obj.order)
                P_ab(k) = sum(Obj.con_mat(k,:)).*sum(Obj.con_mat(:,k));
            end
            P_E = sum(P_ab)/(Obj.Len_data^2);
            K_index = (P_O-P_E)/(1-P_E);            
        end
    end
end