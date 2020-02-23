import numpy as np
import os
from os.path import isfile, join

# np.random.seed(12345)
if (not os.path.isdir('./myData')):
    os.mkdir('./myData')

instituteSize = 20
reviewerSize = 30
numReviewerPerPaper = 2

studentProb = 0.7
studentAcceptableProb = 0.2
nonStudentAcceptableProb = 0.4
highRankProb = 0.5

def write2File(dataPath, dict):
    out = open(dataPath, 'w')
    for key in dict.keys():
        value = dict[key]
        out.write('%s,%s\n' % (key, str(value)))
    out.close()

def writeSet2File(dataPath, set):
    out = open(dataPath, 'w')
    for value in set:
        out.write('%s\n' % value)
    out.close()

################## Generation Procedure #################


# Generate Institutes
instituteSet = set()
idx = 0
while idx < instituteSize:
    key = 'i' + str(idx)
    idx += 1
    instituteSet.add(key)
writeSet2File('./myData/institute.txt', instituteSet)

# Generate Institute Submission
instituteSubmissionDict = {}
paperSize = 0
n = 10
p = 0.5
for institute in instituteSet:
    numSubmission = np.random.binomial(n, p)
    paperSize += numSubmission
    instituteSubmissionDict[institute] = numSubmission

# Generate Authors
authorSet = set()
idx = 0
for idx in range(paperSize):
    key = 'a' + str(idx)
    authorSet.add(key)
writeSet2File('./myData/author.txt', authorSet)

# Generate Papers
paperSet = set()
idx = 0
for idx in range(paperSize):
    key = 'p' + str(idx)
    paperSet.add(key)
writeSet2File('./myData/paper.txt', paperSet)

# Generate Reviewers
reviewerSet = set()
for idx in range(reviewerSize):
    key = 'r' + str(idx)
    reviewerSet.add(key)
writeSet2File('./myData/reviewer.txt', reviewerSet)

# Generate Submits
submitDict = {}
for idx in range(paperSize):
    key = 'p' + str(idx)
    value = 'a' + str(idx)
    submitDict[key] = value
# write2File('./myData/submits.txt', submitDict)
out = open('./myData/submits.txt', 'w+')
for paper_key in submitDict.keys():
    author_key = submitDict[paper_key]
    out.write('%s,%s\n' % (author_key, paper_key))
out.close()


# Generate Affiliations
affiliationDict = {}
authorList = [i for i in range(paperSize)]
np.random.shuffle(authorList)
idx = 0
for institute in instituteSubmissionDict:
    numSubmission = instituteSubmissionDict[institute]
    i = 0
    while i < numSubmission:
        key = 'a' + str(authorList[idx])
        affiliationDict[key] = institute
        idx += 1
        i += 1
write2File('./myData/affiliation.txt', affiliationDict)

# Generate Students
studentDict = {}
# studentSize = paperSize* studentProb
rdm_student = np.random.rand(paperSize)
for idx in range(paperSize):
    key = 'a' + str(idx)
    # if idx < studentSize:
    # 	studentDict[key] = 1
    # else:
    # 	studentDict[key] = 0
    if rdm_student[idx] <= studentProb:
        studentDict[key] = 1
    else:
        studentDict[key] = 0
write2File('./myData/student.txt', studentDict)

# Generate Reviews
reviewDict = {}
idx = 0
for idx in range(paperSize):
    list = np.random.choice(reviewerSize, numReviewerPerPaper, replace=False)
    value = []
    for i in list:
        value.append('r' + str(i))
    key = 'p' + str(idx)
    reviewDict[key] = value
out = open('./myData/reviews.txt', 'w+')
for paper_key in reviewDict.keys():
    for reviewer in reviewDict[paper_key]:
        out.write('%s,%s\n' % (reviewer, paper_key))
out.close()

# Generate Acceptable
acceptableDict = {}
idx = 0
rdm_paper = np.random.rand(paperSize)
for idx in range(paperSize):
    paper_key = 'p' + str(idx)
    author_key = submitDict[paper_key]
    isStudent = studentDict[author_key]
    if (isStudent == 1):
        if (rdm_paper[idx] <= studentAcceptableProb):
            acceptableDict[paper_key] = 1
        else:
            acceptableDict[paper_key] = 0
    else:
        if (rdm_paper[idx] <= nonStudentAcceptableProb):
            acceptableDict[paper_key] = 1
        else:
            acceptableDict[paper_key] = 0
write2File('./myData/acceptable.txt', acceptableDict)

# Generate HighRank Institutes
highRankDict = {}
rdm_institute = np.random.rand(instituteSize)
for idx in range(instituteSize):
    key = 'i' + str(idx)
    if (rdm_institute[idx] < highRankProb):
        highRankDict[key] = 1
    else:
        highRankDict[key] = 0
write2File('./myData/highRank.txt', highRankDict)

# Generate positiveReviews
'''
    Gives a tuple of True/False according to the binary representation
    of the number:

    __numbits(0, 3) -> (False, False, False)
    __numbits(1, 3) -> (False, False, True)
    ...
    __numbits(6, 3) -> (True,  True,  False)
    __numbits(7, 3) -> (True,  True,  True)
'''

def __num2bits(n, nbits):
    bits = [False] * nbits
    for i in range(1, nbits + 1):
        bits[nbits - i] = bool(n % 2)
        n >>= 1
    return tuple(bits)


x1 = 0.6
x2 = 0.9
reviewProbability = [0.15, 0.05, 0.20, 0.15, 0.85, x1, 0.85, x2]

prob_dict = dict()
for i in range(8):
    prob_dict[__num2bits(i, 3)] = reviewProbability[i]

positiveDict = {}
rdm = np.random.rand(paperSize* numReviewerPerPaper)
idx = 0
for paper_key in paperSet:
    author_key = submitDict[paper_key]
    institute_key = affiliationDict[author_key]
    is_high_rank = bool(highRankDict[institute_key])
    is_student = bool(studentDict[author_key])
    is_acceptable = bool(acceptableDict[paper_key])
    prob_key = (is_acceptable, is_high_rank, is_student)

    for reviewer_key in reviewDict[paper_key]:
        key = reviewer_key + ',' + paper_key
        if (rdm[idx] < prob_dict[prob_key]):
            positiveDict[key] = 1
        else:
            positiveDict[key] = 0
        idx += 1
write2File('./myData/positiveReviews.txt', positiveDict)

# Generate Summary
summaryProbability = [0.0, 0.20, 0.20, 0.90]

prob_dict = dict()
for i in range(2**numReviewerPerPaper):
    prob_dict[__num2bits(i, numReviewerPerPaper)] = summaryProbability[i]

summaryDict = {}
rdm_paper = np.random.rand(paperSize)
idx = 0
for paper_key in paperSet:
    key = []
    for reviewer in reviewDict[paper_key]:
        reviewer_key = reviewer+ ','+ paper_key
        review = bool(positiveDict[reviewer_key])
        key.append(review)
    key = tuple(key)
    if (rdm_paper[idx] < prob_dict[key]):
        summaryDict[paper_key] = 1
    else:
        summaryDict[paper_key] = 0
    idx += 1
write2File('./myData/positiveSummary.txt', summaryDict)

