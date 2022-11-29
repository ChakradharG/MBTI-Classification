class Words:
	def __init__(self, string):
		self.words = self.parse(string)	# stores a list of words (tokens) for a particular person
	
	def __repr__(self):
		return str(self.words)
	
	def parse(self, string):
		l = []
		string = string.split('|||')	# different post of a person are separated by |||
		for s in string:
			# removing undesired characters
			s = s.replace("'", '')
			s = s.replace('"', '')
			s = s.replace(',', ' ')
			s = s.replace('...', ' ')
			s = s.split()
			for i in s:
				if i.startswith('http'):	# removing hyperlinks
					continue
				i = i.replace('.', '-')
				i = i.split('-')
				# only storing alphabets in lowercase
				string = ''
				for j in list(map(lambda x: x.lower(), i))[0]:
					string += j if j.isalpha() else ''

				if string != '':
					l.append(string)
		
		return l
