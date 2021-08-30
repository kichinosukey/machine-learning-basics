# machine-learning-basics

outline
1. linear regression    | doc | ex | test
2. logistic regression  | doc | ex | test
3. neural network       | doc | ex | test

# 1. linear regression

## Dataset
- https://archive.ics.uci.edu/ml/machine-learning-databases/housing/

## Attribute Information
    1. CRIM      per capita crime rate by town/町別一人当たりの犯罪率
    2. ZN        proportion of residential land zoned for lots over 
                 25,000 sq.ft./25,000平方フィート以上の土地に指定されている宅地の割合
    3. INDUS     proportion of non-retail business acres per town/町ごとの非小売業の面積の割合
    4. CHAS      Charles River dummy variable (= 1 if tract bounds 
                 river; 0 otherwise)/チャールズ川ダミー変数（＝川に接している場合は1、そうでない場合は0）
    5. NOX       nitric oxides concentration (parts per 10 million)/NOX 一酸化窒素濃度（1,000万分の1）
    6. RM        average number of rooms per dwelling/住戸あたりの平均部屋数
    7. AGE       proportion of owner-occupied units built prior to 1940/1940年以前に建てられた持ち家の割合
    8. DIS       weighted distances to five Boston employment centres/ボストンの5つの雇用センターまでの加重距離
    9. RAD       index of accessibility to radial highways/放射状高速道路へのアクセス性の指標
    10. TAX      full-value property-tax rate per $10,000/1万ドルあたりの固定資産税の全額負担率
    11. PTRATIO  pupil-teacher ratio by town/町ごとの生徒と教師の比率
    12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks 
                 by town/町別の黒人の割合
    13. LSTAT    % lower status of the population/人口の低いステータス[%]
    14. MEDV     Median value of owner-occupied homes in $1000's/持ち家の中央値（1000ドル台）

## Single variable

- Cost Function
- Gradient Descent
  
## Linear algebra basic

- Numpy fundamentals

## Excercise

- Predict house price with single variable

## Multiple variables

- Cost Function
- Feature Normalization
- Gradient Descent

## Excercise

- Predict house price with multiple variables


# 2. logistic regression

## Binary Classification

- Decision Boundary
- Cost Function
- Gradient Descent
- Advanced optimization


## Multiclass Classification

- Cost Function
- Gradient Descent
- Regularization

# 3. Neural Networks

- Cost Function
- Backpropagation
- Gradient Checking
- Random Initialization
- 