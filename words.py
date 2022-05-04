class Words:
	def __init__(self, string):
		self.words = self.parse(string)	# stores a list of words (tokens) for a particular person
	
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
				l.extend(filter(lambda x: x.isalpha(), map(lambda x: x.lower(), i)))
		
		return l
