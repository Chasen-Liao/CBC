% 随机初始化转运方案
function [trans_plan] = Init_trans(order_plan, waste)
trans_plan = zeros(402, 8 * 24);
for i = 1:24
    waste_week = waste(:, i);
    ranp = zeros(8, 1);
    transporter = [];
    for t = 1:8
        if waste_week(t) ~= 0
            transporter = [transporter, t];
        end
    end
    
    for j = 1:402
        if order_plan(j, i) ~= 0
            rand_num = transporter(randperm(numel(transporter), 1)); 
            if ranp(rand_num) + order_plan(j, i) < 6000
                ranp(rand_num) = ranp(rand_num) + order_plan(j, i);
                trans_plan(j, (i - 1) * 8 + rand_num) = order_plan(j, i);
            else
                surplus = ranp(rand_num) + order_plan(j, i) - 6000;
                ranp(rand_num) = 6000;
                trans_plan(j, (i - 1) * 8 + rand_num) = order_plan(j, i) - surplus;
                
                rand_num = transporter(randperm(numel(transporter), 1));
                while ranp(rand_num) + surplus > 6000
                    if ranp(rand_num) < 6000
                        surplus = surplus - (6000 - ranp(rand_num));
                        trans_plan(j, (i - 1) * 8 + rand_num) = (6000 - ranp(rand_num));
                        ranp(rand_num) = 6000;
                    end
                    rand_num = transporter(randperm(numel(transporter), 1)); 
                end
                ranp(rand_num) = ranp(rand_num) + surplus;
                trans_plan(j, (i - 1) * 8 + rand_num) = surplus;
            end
        end
    end
end
end


            
