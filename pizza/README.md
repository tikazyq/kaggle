最近闲来无事在学一些data science的东西，Kaggle正好提供了这样一个可以好（xian）好（de）练（dan）手（teng）的数据。
链接：http://www.kaggle.com/c/random-acts-of-pizza
相关论文：http://cs.stanford.edu/~althoff/raop-dataset/altruistic_requests_icwsm.pdf

数据来自于reddit上发起的一个free pizza competition，用户在上面发帖（post）卖萌请求free pizza，有哭诉母亲生病想吃pizza的，有抱怨自己工资低吃不起饭的。各个pizza提供者根据帖子的信息决定是否给申请人（requester）提供免费pizza。

rbroberg提供了一个简单的解法（用Julia写的）：https://github.com/rbroberg/kaggle.com/blob/master/rand.pizza/scripts/random_pizza_beat_the_benchmark.jl

解法步骤：
用每个document的text content生成一系列words，并去掉stopwords
算出所有words出现的频率（frequency）
根据training data的结果（收到／没收到pizza）以及其frequency算每个word对于收到pizza的probability（称作prob_dict)
根据prob_dict算出每个document对应的收到pizza的平均probability
计算training data所有documents平均probability分布（如下图），发现很cutoff很明显在0.3（也就是说，当document的probability大于0.3时，能收到pizza）
将cutoff>0.3到test data上，得到其预测值
