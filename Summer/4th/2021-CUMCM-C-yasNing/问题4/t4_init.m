%% 初始化订货方案，转运方案，并计算转运后的产能
function [order_plan, trans_plan, capacity] = t4_init(predict_402, predict_waste, class)
    %初始化订货方案
    num = predict_402(:, 1);
    order_plan = zeros(402, 24);
    for i = 1:24
        waste_week = predict_waste(:, i);
        nt = 0;
        for t = 1:8
            if waste_week(t) ~= 0
                nt = nt + 1;
            end
        end
        nt = nt * 6000;
        c = 0;
        for j = 1:402
            n = num(j);
            x = round(predict_402(j, i + 2) * (0.7 + 0.3 * rand()));
            if c + x >= nt
                break;
            end
            order_plan(n, i) = x;
            c = c + order_plan(n, i);
        end
    end

    %初始化转运方案
    trans_plan = Init_trans(order_plan, predict_waste);
    %计算损耗后的每周产能
    capacity = zeros(24, 1);
    for i = 1:24
        for j = 1:402
            wasteage = 0;
            for t = 1:8
                wasteage = wasteage + trans_plan(j, (i - 1) * 8 + t) * predict_waste(t, i) / 100;
            end
            if class(j) == 1.2
                capacity(i) = (order_plan(j, i) - wasteage) / 0.6 + capacity(i);
            elseif class(j) == 1.1
                capacity(i) = (order_plan(j, i) - wasteage) / 0.66 + capacity(i);
            elseif class(j) == 1
                capacity(i) = (order_plan(j, i) - wasteage) / 0.72 + capacity(i);
            end
        end
    end
end
