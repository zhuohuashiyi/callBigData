import sys
import sqlite3
import xlwt
import xlrd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


def demo(X):
	colors = ['b', 'g', 'r']
	markers = ['o', 'v', 's']
	# k means determine k
	distortions = []
	K = range(1, 10)
	for k in K:
		kmeanModel = KMeans(n_clusters=k).fit(X)
		kmeanModel.fit(X)
		distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
	plt.plot(K, distortions, 'bx-')
	plt.xlabel('k')
	plt.ylabel('Distortion')
	plt.title('The Elbow Method showing the optimal k')
	plt.show()

matplotlib.rcParams['font.sans-serif'] = ['SimHei']


# 将txt文件中的数据导入到数据库中
def store_data(str):
	k = 0
	conn = sqlite3.connect("data.db")
	c = conn.cursor()
	c.execute('''CREATE TABLE IF NOT EXISTS data
       (
       DATA_ID        CHAR(8)    NOT NULL,
       CALLING_NBR    VARCHAR(20) NOT NULL,
       CALLED_NBR     VARCHAR(50),
       CALLING_OPTR   CHAR(1),
       CALLED_OPTR    CHAR(1),
       CALLING_CITY   VARCHAR(10),
       CALLED_CITY    VARCHAR(10),
       CALLING_ROAM_CITY  VARCHAR(10),
       CALLED_ROAM_CITY   VARCHAR(10),
       START_TIME      CHAR(8),
       END_TIME        CHAR(8),
       RAW_DUR        BIGINT,
       CALL_TYPE    CHAR(1),
       CALLING_CELL   VARCHAR(10));''')
	conn.commit()
	with open(str, "r") as f:
		for line in f:
			k += 1
			if k % 1000 != 0:
				continue
			list = line.split()
			print(list)
			sql = '''insert into data values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);'''
			c.execute(sql, tuple(list))
			conn.commit()	
	conn.close()


def test():
	conn = sqlite3.connect("data.db")
	c = conn.cursor()
	c.execute('''SELECT DISTINCT CALLING_NBR FROM data;''')
	nbrs = c.fetchall()
	print(len(nbrs))
	conn.close()


def delete():
	conn = sqlite3.connect("data.db")
	c = conn.cursor()
	sql = '''delete from result;'''
	c.execute(sql)
	conn.commit()
	c.execute("delete from proportion;")
	conn.commit()
	c.execute("delete from phone;")
	conn.commit()
	conn.close()


# sql 建立数据库table
def create_table():
	conn = sqlite3.connect("data.db")
	c = conn.cursor()
	c.execute('''CREATE TABLE IF NOT EXISTS result
       (
       CALLING_NBR    VARCHAR(20) NOT NULL,
       MEAN_NUMBER_PER_DAY  BIGINT,
       MEAN_DURATION_PER_DAY  DOUBLE,
       MEAN_DURATION_PER_PHONE  DOUBLE);''')
	conn.commit()
	c.execute('''CREATE TABLE IF NOT EXISTS phone
		(
		CALLING_NBR  VARCHAR(20)  NOT NULL,
		POSITIVE_PHONE_NUMBER BIGINT,
		NEGATIVE_PHONE_NUMBER BIGINT);''')
	conn.commit()
	c.execute('''CREATE TABLE IF NOT EXISTS proportion
       (
       CALLING_NBR    VARCHAR(20) NOT NULL,
       total_time   BIGINT,
       time1  BIGINT,
       time2  BIGINT,
       time3  BIGINT,
       time4  BIGINT,
       time5  BIGINT,
       time6  BIGINT,
       time7  BIGINT,
       time8  BIGINT
       );''')
	conn.commit()
	conn.close()


# 计算数据
def calculate():
	conn = sqlite3.connect("data.db")
	c = conn.cursor()
	# 暂时将跨夜数据抹掉
	c.execute("DELETE FROM DATA WHERE START_TIME > END_TIME;")
	c.execute('''SELECT DISTINCT CALLING_NBR FROM data;''')
	nbrs = c.fetchall()
	print(len(nbrs))
	k = 0
	for nbr in nbrs:
		k += 1
		if k < 8265:
			continue
		phone = nbr[0]
		#print(phone)
		c.execute("SELECT COUNT(*) FROM data WHERE CALLING_NBR = ?;", (phone,))
		res = c.fetchall()
		phone_num = res[0][0]
		#print(phone_num)
		c.execute("SELECT COUNT(*) FROM (SELECT DISTINCT DATA_ID FROM data WHERE CALLING_NBR = ?);", (phone,))
		res = c.fetchall()
		day_num = res[0][0]
		#print(day_num)
		c.execute("SELECT SUM(RAW_DUR) FROM DATA WHERE CALLING_NBR = ?;", (phone,))
		res = c.fetchall()
		total_time = res[0][0]
		#print(total_time)
		c.execute("INSERT INTO result VALUES(?, ?, ?, ?);", (phone, phone_num / day_num, total_time / day_num, 
			total_time / phone_num))
		conn.commit()
		c.execute("SELECT START_TIME, END_TIME FROM DATA WHERE CALLING_NBR = ?", (phone, ))
		time = [0, 0, 0, 0, 0, 0, 0, 0]
		base = ["00:00:00", "03:00:00", "06:00:00", "09:00:00", "12:00:00", "15:00:00", "18:00:00", "21:00:00", "24:00:00"]
		begin = -1
		end = -1
		for each in c.fetchall():
			for i in range(8):
				if base[i] <= each[0] and base[i + 1] >= each[0]:
					begin = i
					
				if base[i] <= each[1] and base[i + 1] >= each[1]:
					end = i
					
		print(begin)
		print(end)
		if begin <= end:
			for i in range(begin, end + 1):
				time[i] += 1
		else:
			for i in range(end, 8):
				time[i] += 1
			for i in range(0, begin + 1):
				time[i] += 1
		s = 0
		for i in range(8):
			s += time[i]
		c.execute("INSERT INTO proportion VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
			(phone, s, time[0], time[1], time[2], time[3], time[4], time[5], time[6], time[7]))
		conn.commit()
		c.execute('''SELECT COUNT(*) FROM DATA WHERE CALLING_NBR = ?;''', (phone, ))
		t9 = c.fetchall()[0][0]
		c.execute('''SELECT COUNT(*) FROM DATA WHERE CALLED_NBR = ?;''', (phone, ))
		t10 = c.fetchall()[0][0]
		c.execute('''INSERT INTO phone VALUES(?, ?, ?)''', (phone, t9, t10))
		conn.commit()
	conn.close()


# 将数据库中的数据保存为xls文件
def transfer_to_xls():
	workbook = xlwt.Workbook()
	sheet1 = workbook.add_sheet("result")
	names = ['主叫号码', '每日平均通话次数', '每日平均通话时长', '每个通话平均时长']
	for i in range(4):
		sheet1.write(0, i, names[i])
	conn = sqlite3.connect("data.db")
	c = conn.cursor()
	c.execute("SELECT * FROM result;")
	res = c.fetchall()
	for i in range(len(res)):
		for j in range(4):
			sheet1.write(i + 1, j, res[i][j])
	sheet2 = workbook.add_sheet("proportion")
	names_1 = ['主叫号码', '时间段1占比', '时间段2占比', '时间段3占比', '时间段4占比', '时间段5占比', 
		'时间段6占比', '时间段7占比', '时间段8占比']
	for i in range(9):
		sheet2.write(0, i, names_1[i])
	c.execute("SELECT * FROM proportion;")
	res = c.fetchall()
	for i in range(len(res)):
		if res[i][1] == 0:
			print(res[i])
		for j in range(9):
			if j == 0:
				sheet2.write(i + 1, j, res[i][j])
			else:
				sheet2.write(i + 1, j, res[i][j + 1] / res[i][1])
	workbook.save("result.xls")


def kmeans_xufive(k, ds):
	"""k-means聚类算法
		k       - 指定分簇数量
		ds      - ndarray(m, n)，m个样本的数据集，每个样本n个属性值
	"""
	m, n = ds.shape # m：样本数量，n：每个样本的属性值个数
	result = np.empty(m, dtype=np.int) # m个样本的聚类结果
	cores = ds[np.random.choice(np.arange(m), k, replace=False)] # 从m个数据样本中不重复地随机选择k个样本作为质心
	while True: # 迭代计算
		d = np.square(np.repeat(ds, k, axis=0).reshape(m, k, n) - cores)
		distance = np.sqrt(np.sum(d, axis=2)) # ndarray(m, k)，每个样本距离k个质心的距离，共有m行
		index_min = np.argmin(distance, axis=1) # 每个样本距离最近的质心索引序号
		if (index_min == result).all(): # 如果样本聚类没有改变
			return result, cores # 则返回聚类结果和质心数据
		result[:] = index_min # 重新分类
		for i in range(k): # 遍历质心集
			items = ds[result==i] # 找出对应当前质心的子样本集
			cores[i] = np.mean(items, axis=0) # 以子样本集的均值作为当前质心的位置


def draw_first_bar(results, cores):
	num = [0, 0, 0, 0]
	order = cores
	order.sort()
	print(order)
	pos = []
	for core in cores:
		for i in range(4):
			if order[i] == core:
				pos.append(i)
				break
	for result in results:
		num[pos[result]] += 1
	print(num)
	label_list = ["每日通话时长较短", "每日通话时长适中", "每日通话时长较长", "每日通话时长很长"]
	num_list = [10, 25, 40, 55]
	x = range(4)
	rects1 = plt.bar(x, num, width=0.4, alpha=0.8, color='red')
	plt.ylim(0, 100)
	plt.ylabel("人数")
	plt.xticks([index + 0.2 for index in range(4)], label_list)
	plt.xlabel("通话时长特征")
	plt.title("根据通话时长特征分类")
	plt.show()


def draw_first_pie(results, cores):
	num = [0, 0, 0, 0]
	order = cores
	order.sort()
	print(order)
	pos = []
	sum = len(results)
	for core in cores:
		for i in range(4):
			if order[i] == core:
				pos.append(i)
				break
	for result in results:
		num[pos[result]] += 1
	print(num)
	label_list = ["每日通话时长较短", "每日通话时长适中", "每日通话时长较长", "每日通话时长很长"]
	size = [num[0] / sum, num[1] / sum, num[2] / sum, num[3] / sum]    # 各部分大小
	color = ["red", "green", "blue", "yellow"]     # 各部分颜色
	explode = [0.05, 0, 0, 0.01]   # 各部分突出值
	patches, l_text, p_text = plt.pie(size, explode=explode, colors=color, labels=label_list, labeldistance=1.1, autopct="%1.1f%%", shadow=False, startangle=90, pctdistance=0.6)
	plt.axis("equal")    # 设置横轴和纵轴大小相等，这样饼才是圆的
	plt.legend()
	plt.show()


def draw_second_bar(results, cores):
	num = [0, 0, 0, 0, 0, 0]
	for result in results:
		num[result] += 1
	label_list = ["正常时段通话但时长较长", "正常时段通话但时长较短", "主要深夜通话"]
	x = range(3)
	rects1 = plt.bar(x, num, width=0.4, alpha=0.8, color='red')
	plt.ylim(0, 100)
	plt.ylabel("人数")
	plt.xticks([index + 0.2 for index in range(3)], label_list)
	plt.xlabel("通话时间段特征")
	plt.title("根据通话时间段特征分类")
	plt.show()


def draw_second_pie(results, cores):
	num = [0, 0, 0, 0, 0, 0]
	for result in results:
		num[result] += 1
	sum = 0
	for each in num:
		sum += each
	label_list = ["正常时段通话但时长较长", "正常时段通话但时长较短", "主要深夜通话"]
	size = [num[0] / sum, num[1] / sum, num[2] / sum]    # 各部分大小
	color = ["red", "green", "blue"]     # 各部分颜色
	explode = [0.05, 0, 0]   # 各部分突出值
	patches, l_text, p_text = plt.pie(size, explode=explode, colors=color, labels=label_list, labeldistance=1.1, autopct="%1.1f%%", shadow=False, startangle=90, pctdistance=0.6)
	plt.axis("equal")    # 设置横轴和纵轴大小相等，这样饼才是圆的
	plt.legend()
	plt.show()


if __name__ == "__main__":
	#test()
	#delete()
	#store_data("data.txt")
	#create_table()
	#calculate()
	transfer_to_xls()

	#create_table_result()
	'''
	conn = sqlite3.connect("data.db")
	c = conn.cursor()
	c.execute("SELECT * FROM proportion;")
	res = c.fetchall()
	ds = np.asarray([res[i][2:] for i in range(len(res))])
	print(ds.shape)
	#demo(ds)
	results, cores = kmeans_xufive(6, ds)
	print(results)
	print(cores)
	draw_second_pie(results, [core[0] for core in cores])
	draw_second_bar(results, [core[0] for core in cores])
	#calculate() '''
	#demo(ds)'''
	
	
	

