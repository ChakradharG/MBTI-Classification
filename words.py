import re



class Words:
	def __init__(self, string):
		self.words = self.parse(string)
	
	def parse(self, string):
		# change to regex
		l = []
		for i in string.split():
			for j in i.split('|||'):
				if j.startswith("'http") or j.startswith('http'):
					continue
				if j.startswith("'"):
					l.append(j[1:].lower())
				elif j.endswith("'"):
					l.append(j[:-1].lower())
				elif j.startswith("..."):
					l.append(j[3:].lower())
				elif j.endswith("..."):
					l.append(j[:-3].lower())
				else:
					l.append(j.lower())
		
		return l
