def fall_into_cliff(row, col):
	if(row == 3):
		if(col > 0 and col < 11):
			return 1
	else:
		return 0

class Env:

	def __init__(self, row, col):
		self.pos_row = row
		self.pos_col = col
		self.state = self.pos_row * 12 + self.pos_col

	def getStateAction(self, state):
		# 左上角
		if state == 0:
			return [1, 3]
		# 右上角
		if state == 11:
			return [1, 2]
		# 左下角
		if state == 36:
			return [0, 3]
		# 右下角
		if state == 47:
			return [0, 2]
		# 左边界
		if state in [12, 24]:
			return [0, 1, 3]
		# 右边界
		if state in [23, 35]:
			return [0, 1, 2]
		# 上边界
		if state in range(1, 11):
			return [1, 2, 3]
		# 下边界
		if state in range(37, 47):
			return [0, 2, 3]

		return [0, 1, 2, 3]

	def getPosition(self):
		return self.state

	def transition(self, action):
		if(action < 2):
			if(action == 0):
				# 0 向上
				self.pos_row = self.pos_row - 1 if self.pos_row > 0 else self.pos_row
			else:
				# 1 向下
				self.pos_row = self.pos_row + 1 if self.pos_row < 3 else self.pos_row
		else:
			if(action == 2):
				# 2 向左
				self.pos_col = self.pos_col - 1 if self.pos_col > 0 else self.pos_col
			else:
				# 3 向右
				self.pos_col = self.pos_col + 1 if self.pos_col < 11 else self.pos_col

		self.state = self.pos_row * 12 + self.pos_col
		if(fall_into_cliff(self.pos_row, self.pos_col)):
			return self.state, -100

		return self.state, -1

	def reset(self):
		self.pos_row = 3
		self.pos_col = 0
		self.state = self.pos_row * 12 + self.pos_col
