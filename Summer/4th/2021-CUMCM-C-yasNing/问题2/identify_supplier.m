%% 确定供应商数量
suppliers = xlsread('..\问题1\问题1.xlsx', '供应商重要性', 'A2:A403');
% 满足每周产能需求的供应商数量
number = zeros(240, 1);
% 每个供货商每周的供货量
supply_week = xlsread('..\附件1 近5年402家供应商的相关数据.xlsx', '供应商的供货量（m³）', 'B2:IH403');

%库存产能
capacity = 0;
Min = 30;
Max = 50;
below = 0;
upper = 0;
for i = 2:241
    count = 1;
    while count < Min
        if supply_week(suppliers(count), 1) == 1.2
            capacity = supply_week(suppliers(count), i) / 0.6 + capacity;
        elseif supply_week(suppliers(count), 1) == 1.1
            capacity = supply_week(suppliers(count), i) / 0.66 + capacity;
        elseif supply_week(suppliers(count), 1) == 1
            capacity = supply_week(suppliers(count), i) / 0.72 + capacity;
        end
        count = count + 1;
    end

    while capacity < 2.82 * 1e4 * 2 
        if supply_week(suppliers(count), 1) == 1.2
            capacity = supply_week(suppliers(count), i) / 0.6 + capacity;
        elseif supply_week(suppliers(count), 1) == 1.1
            capacity = supply_week(suppliers(count), i) / 0.66 + capacity;
        elseif supply_week(suppliers(count), 1) == 1
            capacity = supply_week(suppliers(count), i) / 0.72 + capacity;
        end

        count = count + 1;
        if count > Max
            break;
        end
    end

    if count > Max
        below = below + 1;
        while capacity < 2.82 * 1e4
            if supply_week(suppliers(count), 1) == 1.2
                capacity = supply_week(suppliers(count), i) / 0.6 + capacity;
            elseif supply_week(suppliers(count), 1) == 1.1
                capacity = supply_week(suppliers(count), i) / 0.66 + capacity;
            elseif supply_week(suppliers(count), 1) == 1
                capacity = supply_week(suppliers(count), i) / 0.72 + capacity;
            end
            if count == 402
                break;
            else
                count = count + 1;
            end
        end
    else
        upper = upper + 1;
    end
    

    capacity = capacity - 2.82 * 1e4;
    number(i - 1) = count;
end
prop_below = below / 240 * 100;
prop_upper = upper / 240 * 100;
%% 绘图
x = 1:240;
figure, plot(x, number, '-b', 'linewidth', 2);
hold on;
grid minor;
title('确定供应商数量');
xlabel('周数');
ylabel('供应商数量');




