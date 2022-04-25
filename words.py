class Words:
	def __init__(self, string):
		self.words = self.parse(string)
	
	def parse(self, string):
		l = []
		string = string.split('|||')
		for s in string:
			s = s.replace("'", '')
			s = s.replace('"', '')
			s = s.replace(',', ' ')
			s = s.replace('...', ' ')
			s = s.split()
			for i in s:
				if i.startswith('http'):
					continue
				i = i.replace('.', '-')
				i = i.split('-')
				l.extend(filter(lambda x: x.isalpha(), map(lambda x: x.lower(), i)))
		
		return l
