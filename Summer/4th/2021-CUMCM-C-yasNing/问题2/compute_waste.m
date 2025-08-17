%% 计算损耗=损耗量*材料单价
function [W] = compute_waste(plan, waste, class)
    W = 0;
    for i = 1:24
        for j = 1:402
            wasteage = 0;
            for t = 1:8
                wastage = wasteage + plan(j, (i - 1) * 8 + t) * waste(t, i) / 100;
            end
            W = W + wastage * class(j);
        end
    end        
end
