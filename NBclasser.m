Train= readtable('titanic\train.csv');%通过从文件中读取列向数据来创建表
Test = readtable('titanic\test.csv');

%查看缺少情况
Train.Fare(Train.Fare == 0) = NaN;      % treat 0 fare as NaN
vars = Train.Properties.VariableNames;  % extract column names提取列名
figure
imagesc(ismissing(Train))%ismissing返回逻辑数组，指示数组或表中的哪些元素包含缺失值 imagesc使用缩放颜色显示图像
set(gca,'XTick', 1:12,'XTickLabel',vars);%gca获取当前figure的句柄,
xtickangle(-45)%旋转 x 轴刻度标签,负值表示顺时针旋转

%补充出发港口缺少值
% get most frequent value 获取最频繁值
disp(grpstats(Train(:,{'Survived','Embarked'}), 'Embarked'))

% apply it to missling value
for i = 1 : 891
    if isempty(Train.Embarked{i})
        Train.Embarked{i}='S';%用出舱口最多的填充
    end
end
 
for i = 1 : 418
    if isempty(Test.Embarked{i})
        Test.Embarked{i}='S';
    end
end

for i=1:891
    if strcmp(Train.Embarked{i} ,'S')
        Train.Embarked{i}=1;
    elseif strcmp(Train.Embarked{i} ,'C')
        Train.Embarked{i}=2;
    else
        Train.Embarked{i}=3;
    end
end
for i=1:418
    if strcmp(Test.Embarked{i} ,'S')
        Test.Embarked{i}=1;
    elseif strcmp(Test.Embarked{i} ,'C')
        Test.Embarked{i}=2;
    else
        Test.Embarked{i}=3;
    end
end
% convert the data type from categorical to double，将数据类型从分类转换为双精度
Train.Embarked = double(cell2mat(Train.Embarked));%将元胞数组转换为基础数据类型的普通数组
Test.Embarked = double(cell2mat(Test.Embarked));
disp(grpstats(Train(:,{'Survived','Embarked'}), 'Embarked'))

% change Sex to tpye "double"
for i = 1 : 891
    if strcmp(Train.Sex{i} ,'male')%比较字符串,如果二者相同,则返回1(true)
        Train.Sex{i}=1;
    else
        Train.Sex{i}=2;
    end
end
for i = 1 : 418
    if strcmp(Test.Sex{i} ,'male')
        Test.Sex{i}=1;
    else
        Test.Sex{i}=2;
    end
end
Train.Sex = cell2mat(Train.Sex);
Test.Sex = cell2mat(Test.Sex);
disp(grpstats(Train(:,{'Survived','Sex'}), 'Sex'))%grpstats分组平均函数

for i=1:891 
    if Train.SibSp(i)<1
        Train.SibSp(i)=1;
    elseif Train.SibSp(i)<3
        Train.SibSp(i)=2; 
    else
        Train.SibSp(i)=3; 
    end
end
for i=1:418 
    if Test.SibSp(i)<1
        Test.SibSp(i)=1;
    elseif Test.SibSp(i)<3
        Test.SibSp(i)=2; 
    else
        Test.SibSp(i)=3; 
    end
end
disp(grpstats(Train(:,{'Survived','SibSp'}), 'SibSp'))

for i=1:891 
    if Train.Parch(i)<1
        Train.Parch(i)=1;
    elseif Train.Parch(i)<4
        Train.Parch(i)=2; 
    else
        Train.Parch(i)=3; 
    end
end
for i=1:418 
    if Test.Parch(i)<1
        Test.Parch(i)=1;
    elseif Test.Parch(i)<4
        Test.Parch(i)=2; 
    else
        Test.Parch(i)=3; 
    end
end
disp(grpstats(Train(:,{'Survived','Parch'}), 'Parch'))

%删掉一些列
Train(:,{'Name','Ticket','Fare','Cabin','Age'}) = [];
Test(:,{'Name','Ticket','Fare','Cabin','Age'}) = [];
data = Train.Variables;
t = data(:,3:7);
c = data(:,2);
X = t';
y = (c+1)';
idx = randperm(891);%返回行向量,其中包含从1到n没有重复元素的整数随机排列
X_train = X(:,idx);
y_train = y(idx);

data1 = Test.Variables;
t1 = data1(:,3:6);
X1 = t1';
idx1 = randperm(418);
X_test = X1(:,idx1);
[pw,c,a_len] = my_nb1(X_train,y_train);

[post_p,test_lab] = my_testnb1(X_test,pw,c,a_len,2,4);
d = (test_lab-1)';
data2 = [Test.PassengerId,d]; 
dlmwrite('E:\MATLAB R2021a\matlab\bin\submission.csv',data2)

function [pw,c,a_len] = my_nb1(x,gnd)
[row,~] = size(x);
my_lab = unique(gnd);%返回的是和A中一样的值，但是没有重复元素。产生的结果向量按升序排序
numClass = length(my_lab);
for i = 1 : numClass%对gnd中的不重复元素求平均值
    pw(i)= sum(gnd==my_lab(i));
    pw(i) = pw(i)/length(gnd);%死亡概率，生还概率
end
for i = 1 : row
    a = unique(x(i,:));
    a_len(i) = length(a);
    for j = 1:numClass
        temp = x(i,gnd==my_lab(j));  
        b = length(temp);
        for k=1:a_len(i)
            num1= temp==a(k);
            num1_len=sum(num1);
            c(i,k+a_len(i)*(j-1))=num1_len/b;
        end        
    end    
end
end

function [post_p,test_lab] = my_testnb1(xt,pw,c,a_len,numClass,numVar)

[~,len] = size(xt);
for k = 1 : len
    temp = xt(:,k);
    for i = 1: numClass
        prod = 1;
        for j = 1:numVar
            prod=prod*c(j,temp(j)+a_len(j)*(i-1));
        end   
        post_p(k,i) = prod*pw(i);
    end
    [~,inx] = max(post_p(k,:));
    test_lab(k) = inx;
end
end
