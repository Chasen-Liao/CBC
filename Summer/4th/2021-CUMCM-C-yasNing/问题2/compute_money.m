%% 计算采购成本
function [money, money_all] = compute_money(plan, class)
    money = zeros(24, 1);
    load('weight.mat');
    money_all = 0;
    %计算每周ABC三种原材料的量
    for i = 1:24
        A = 0; 
        B = 0;
        C = 0;
        for j = 1:402
            if class(j) == 1.2
                A = A + plan(j, i);
            elseif class(j) == 1.1
                B = B + plan(j, i);
            elseif class(j) == 1
                C = C + plan(j, i);
            end
        end
        money(i) = A * 1.2 + B * 1.1 + C * 1;
        money_all = money_all + money(i) * weight(i);
    end
end
