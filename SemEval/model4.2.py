import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class WordByWord(nn.Module):
	def __init__(self, vocabSize, maxQuestionLength, maxAnswerLength, gaussianStd = 10, slidingWindowSize = 60):
		self.wordEmbedding = nn.Embedding(vocabSize, 300) #t_k
		self.textLinear = nn.Linear(300, 300) # D=300
		self.questionLinear = nn.Linear(300, 300) # D=300
		self.answerLinear = nn.Linear(300, 300) # D=300
		self.activation = nn.LeakyReLU()
		self.maxQuestionLength = maxQuestionLength
		self.maxAnswerLength = maxAnswerLength
		self.qWeight = nn.Linear(maxQuestionLength, 1, bias = False)
		self.aWeight = nn.Linear(maxAnswerLength, 1, bias = False)
		self.tSlidingWeight = nn.Parameter(torch.normal(torch.zeros(slidingWindowSize, 1), gaussianStd), requires_grad = True)
		self.alphaQ = nn.Parameter(torch.randn(1, 1), requires_grad=True)
		self.alphaA = nn.Parameter(torch.randn(1, 1), requires_grad=True)
		self.alphaQA = nn.Parameter(torch.randn(1, 1), requires_grad=True)
	
	def getCosineSimilarityMatrix(self, a, b):
    	# a: B x dim1 x 300d
    	# b: B x dim2 x 300d
		# output: B x dim1 x dim2
		result = []

		dim1 = a.size(1)
		dim2 = b.size(1)
		for a_ in torch.split(a, 1, dim = 1):
			for b_ in torch.split(b, 1, dim = 1):
				result.append(F.cosine_similarity(a_, b_, dim = 2))

		result = torch.stack(result)
		result = result.view(result.size(1), dim1, dim2)
		result = result.permute(2, 1, 0)
		return result
	
	def weightedMean(self, weights, _input):
    	# weights: nn.Linear(dim, 1)
		# inputs: B x dim
		# output: B x 1
		return weights(_input)/(torch.sum(weights.weight) + 0.0000001)

	def forward(self, question, text, answer):
    	# 4.2.1 Sentential
		questionEmbedding = self.wordEmbedding(question)
		textEmbedding = self.wordEmbedding(text)
		answerEmbedding = self.wordEmbedding(answer)

		questionEncoding = self.activation(self.questionLinear(questionEmbedding))
		textEncoding = self.activation(self.textLinear(textEmbedding))
		answerEncoding = self.activation(self.answerLinear(answerEmbedding))

		tqSimilarity = self.getCosineSimilarityMatrix(textEncoding, questionEncoding) #c^q_{km}. output size: B * text_size * question_size
		taSimilarity = self.getCosineSimilarityMatrix(textEncoding, answerEncoding) #c^a_{kn}. output size: B * text_size * answer_size

		maxQSimilarity = torch.max(tqSimilarity, 1) 	# output size: B * question_size
		maxASimilarity = torch.max(taSimilarity, 1)		# output size: B * answer_size

		Mq = self.weightedMean(self.qWeight, maxQSimilarity) # output size: B * 1
		Ma = self.weightedMean(self.aWeight, maxASimilarity) # output size: B * 1

		Mword = self.alphaQ * Mq + self.alphaA * Ma + self.alphaQA * Mq * Ma # output size: B x 1

		# 4.2.2 Sequential Sliding Window
		qsSimilarity = self.tSlidingWeight.unsqueeze(-1) * questionEncoding.permute(1, 0, 2) 
		qsSimilarity = qsSimilarity.permute(1, 0, 2)  # output size: B * text_size * question_size
		maxQsSimilarity = torch.max(qsSimilarity, 1) 	# output size: B * question_size
		Msq = self.weightedMean(self.qWeight, maxQsSimilarity) # output size: B * 1
		Mssw = self.alphaQ * Msq + self.alphaA * Ma + self.alphaQA * Msq * Ma # output size: B x 1

		return Mword, Mssw