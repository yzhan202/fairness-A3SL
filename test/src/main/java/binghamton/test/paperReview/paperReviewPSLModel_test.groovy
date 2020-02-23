package binghamton.test.paperReview;

import java.text.DecimalFormat;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.umd.cs.psl.application.inference.*;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE;
import edu.umd.cs.psl.application.util.GroundKernels
import edu.umd.cs.psl.application.util.Grounding
import edu.umd.cs.psl.application.learning.weight.em.HardEM
import edu.umd.cs.psl.application.learning.weight.em.PairedDualLearner;

import edu.umd.cs.psl.config.*;
import edu.umd.cs.psl.core.*;
import edu.umd.cs.psl.core.inference.*;
import edu.umd.cs.psl.database.*;
import edu.umd.cs.psl.database.loading.Inserter
import edu.umd.cs.psl.database.rdbms.*;
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver;
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type;
import edu.umd.cs.psl.evaluation.result.*;
import edu.umd.cs.psl.evaluation.statistics.*;
import edu.umd.cs.psl.groovy.*;
import edu.umd.cs.psl.groovy.PSLModel;

import edu.umd.cs.psl.model.argument.ArgumentType;
import edu.umd.cs.psl.model.atom.*;
import edu.umd.cs.psl.model.formula.*;
import edu.umd.cs.psl.model.function.*;
import edu.umd.cs.psl.model.kernel.*;
import edu.umd.cs.psl.model.kernel.linearconstraint.GroundLinearConstraint
import edu.umd.cs.psl.model.predicate.*;
import edu.umd.cs.psl.model.term.*;
import edu.umd.cs.psl.model.rule.*;
import edu.umd.cs.psl.model.weight.*;
import edu.umd.cs.psl.reasoner.Reasoner
import edu.umd.cs.psl.reasoner.ReasonerFactory
import edu.umd.cs.psl.reasoner.admm.ADMMReasonerFactory
import edu.umd.cs.psl.reasoner.function.FunctionComparator
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.parameters.PositiveWeight
import edu.umd.cs.psl.model.parameters.Weight

import edu.umd.cs.psl.ui.loading.*;
import edu.umd.cs.psl.util.database.*;

import com.google.common.collect.Iterables;

import edu.umd.cs.psl.util.database.Queries;

import edu.umd.cs.psl.evaluation.resultui.printer.*;

import java.io.*;
import java.util.*;

import groovy.time.*;

import binghamton.util.FoldUtils
import binghamton.util.GroundingWrapper
import binghamton.fairnessInference.FairnessMPEInference
import binghamton.util.AUCcalculator
import binghamton.util.DataOutputter;


// Config manager
ConfigManager cm = ConfigManager.getManager();
ConfigBundle config = cm.getBundle("paperReview-model");
Logger log = LoggerFactory.getLogger(this.class);

// Databse
def defaultPath = System.getProperty("java.io.tmpdir")
String dbpath = config.getString("dbpath", defaultPath + File.separator + "paperReview-model")
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), config)


def fold = 'bias6';
def dataDir = 'data/biasedPaperReview/'+ fold;


boolean FairnessSignal = false;
double delta = 0.1; //0.01;

PSLModel model = new PSLModel(this, data)

// Default
model.add predicate: "author", types: [ArgumentType.UniqueID]
model.add predicate: "paper", types: [ArgumentType.UniqueID]
model.add predicate: "reviewer", types: [ArgumentType.UniqueID]
// Feature
model.add predicate: "student", types: [ArgumentType.UniqueID]
model.add predicate: "reviews", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "submits", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "acceptable", types: [ArgumentType.UniqueID]
// Target
model.add predicate: "positiveReviews", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "positiveSummary", types: [ArgumentType.UniqueID]
// sensitive attribute
model.add predicate: "institute", types: [ArgumentType.UniqueID]
model.add predicate: "highRank", types: [ArgumentType.UniqueID]
model.add predicate: "affiliation", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

def closedPredicates = [author, paper, reviewer, student, acceptable,
	reviews, submits, institute, highRank, affiliation] as Set;
def inferredPredicates = [positiveReviews, positiveSummary] as Set;

def predicateFileMap = [
	((Predicate)author):"author.txt",
	((Predicate)paper):"paper.txt",
	((Predicate)reviewer):"reviewer.txt",
	((Predicate)student):"student.txt",
	((Predicate)reviews):"reviews.txt",
	((Predicate)submits):"submits.txt",
	((Predicate)positiveReviews):"positiveReviews.txt",
	((Predicate)acceptable):"acceptable.txt",
	((Predicate)positiveSummary):"positiveSummary.txt",
	
	((Predicate)institute):"institute.txt",
	((Predicate)highRank):"highRank.txt",
	((Predicate)affiliation):"affiliation.txt"]

def predicateSoftTruthMap = [
	((Predicate)student):true,
	((Predicate)acceptable):true,
	((Predicate)author):false,
	((Predicate)paper):false,
	((Predicate)reviewer):false,
	((Predicate)reviews):false,
	((Predicate)submits):false,
	((Predicate)positiveReviews):true,
	((Predicate)positiveSummary):true,
	
	((Predicate)institute):false,
	((Predicate)highRank):true,
	((Predicate)affiliation):false]


def initWeight = 5.0

model.add(rule: (acceptable(P) & reviews(R,P)) >> positiveReviews(R,P), squared:true, weight:initWeight)
model.add(rule: (~acceptable(P) & reviews(R,P)) >> ~positiveReviews(R,P), squared:true, weight:initWeight)
model.add(rule: (student(A) & submits(A,P) & reviews(R,P)) >> ~positiveReviews(R,P), squared:true, weight:initWeight)
model.add(rule: (~student(A) & submits(A,P) & reviews(R,P)) >> positiveReviews(R,P), squared:true, weight:initWeight)
model.add(rule: (positiveReviews(R1,P) & positiveReviews(R2,P)) >> positiveSummary(P), squared:true, weight:initWeight)
model.add(rule: (reviews(R,P) & ~positiveReviews(R,P)) >> ~positiveSummary(P), squared:true, weight:initWeight)

if (!FairnessSignal) { 	// Sensitive
	model.add(rule: (student(A) & submits(A,P) & affiliation(A,I) & highRank(I) & reviews(R,P)) >> positiveReviews(R,P), squared:true, weight:initWeight)
//	model.add(rule: (student(A) & submits(A,P) & affiliation(A,I) & ~highRank(I) & reviews(R,P)) >> ~positiveReviews(R,P), squared:true, weight:initWeight)
}

/*
 * Load Weights
 */
BufferedReader reader = null;
def file;
if (!FairnessSignal)
	file = '/baisedBaselineModel.txt'
else
	file = '/fairnessBaselineModel.txt'
reader = new BufferedReader(new FileReader('result/paperReview/bias1'+ file));
String read = null;
List<Double> weights = new ArrayList<Double>();
while ((read = reader.readLine()) != null) {
	def value = Double.valueOf(read)
	weights.add(value);
}
reader.close();

int idx = 0;
for (CompatibilityKernel k : Iterables.filter(model.getKernels(), CompatibilityKernel.class)) {
	k.setWeight(new PositiveWeight(weights.get(idx)));
	idx++;
}


Partition trainPart = new Partition(0)
Partition truthPart = new Partition(1)
Partition inferenceWritePart = new Partition(3)

def inserter;
for (Predicate p: closedPredicates) {
	String fileName = predicateFileMap[p];
	inserter = data.getInserter(p, trainPart);
	def fullFilePath = dataDir+ '/'+ fileName;
	println p.toString()
	if (predicateSoftTruthMap[p]) {
		InserterUtils.loadDelimitedDataTruth(inserter, fullFilePath, ',');
	} else {
		InserterUtils.loadDelimitedData(inserter, fullFilePath, ',');
	}
}

for (Predicate p: inferredPredicates) {
	String fileName = predicateFileMap[p];
	inserter = data.getInserter(p, truthPart);
	def fullFilePath = dataDir+ '/'+ fileName;
	println p.toString()
	if (predicateSoftTruthMap[p]) {
		InserterUtils.loadDelimitedDataTruth(inserter, fullFilePath, ',');
	} else {
		InserterUtils.loadDelimitedData(inserter, fullFilePath, ',');
	}
}

void populateDatabase(DataStore data, Database dbToPopulate, Partition populatePartition, Set inferredPredicates){
	Database populationDatabase = data.getDatabase(populatePartition, inferredPredicates);
	DatabasePopulator dbPop = new DatabasePopulator(dbToPopulate);

	for (Predicate p : inferredPredicates){
		dbPop.populateFromDB(populationDatabase, p);
	}
	populationDatabase.close();
}


Database inferredDB = data.getDatabase(inferenceWritePart, closedPredicates, trainPart);
populateDatabase(data, inferredDB, truthPart, inferredPredicates);

Database wlTruthDB = data.getDatabase(truthPart, inferredPredicates);



Map<String, Integer> truthMap = new HashMap<>();
String fileName = dataDir+ '/positiveSummary.txt'
reader = new BufferedReader(new FileReader(fileName));
read = null;
while((read = reader.readLine()) != null) {
	String[] splited = read.split(",");
	int value = Integer.parseInt(splited[1]);
	truthMap.put(splited[0], value);
}

Map<String, Integer> posReviewTruthMap = new HashMap<>();
fileName = dataDir+ '/positiveReviews.txt'
reader = new BufferedReader(new FileReader(fileName))
read = null
while((read = reader.readLine()) != null) {
	String[] splited = read.split(",");
	int value = Integer.parseInt(splited[2]);
	posReviewTruthMap.put((splited[0]+","+splited[1]), value);
}

// Get Sensitive Attributes
def highRankInstitutes = new HashSet<>();
def instituteMap = new HashMap<>();
def studentSet = new HashSet<>();
Map<String, String> paperAuthorMap = new HashMap<>()
Set<String> protectedGroup = new HashSet<>();
//highRank
fileName = dataDir+ '/highRank.txt';
reader = null;
reader = new BufferedReader(new FileReader(fileName));
read = null;
while((read = reader.readLine()) != null) {
	String[] splited = read.split(",");
	int value = Integer.parseInt(splited[1]);
	if (value == 1) {
		highRankInstitutes.add(splited[0]);
	}
}

fileName = dataDir+ '/student.txt';
reader = null;
reader = new BufferedReader(new FileReader(fileName));
read = null;
while((read = reader.readLine()) != null) {
	String[] splited = read.split(',');
	int value = Integer.parseInt(splited[1]);
	if (value == 1)
		studentSet.add(splited[0]);
}

// affiliation
fileName = dataDir+ '/affiliation.txt';
reader = new BufferedReader(new FileReader(fileName));
read = null;
while((read = reader.readLine()) != null) {
	String[] splited = read.split(",");
	String s = splited[0];
	String i = splited[1];
	instituteMap.put(s, i);
}

// submission
fileName = dataDir+ '/submits.txt'
reader = new BufferedReader(new FileReader(fileName));
read = null;
while((read = reader.readLine()) != null) {
	String[] splited = read.split(',');
	String s = splited[0];
	String p = splited[1];
	paperAuthorMap.put(p, s);
}

int protectedNum=0;
int unprotectedNum = 0;

int posProtectedNum = 0;
int posUnprotectedNum = 0;
int negProtectedNum = 0;
int negUnprotectedNum = 0;

for (Map.Entry<String, String> entry : instituteMap.entrySet()) {
	String stud = entry.getKey();
	String inst = entry.getValue();
	if (studentSet.contains(stud)) {
		if (highRankInstitutes.contains(inst)) {
			unprotectedNum++; // Unprotected Group
		}
		else {
			protectedNum++; // Protected Group
			protectedGroup.add(stud);
		}
	} else {
		unprotectedNum++;
	}
}

System.out.println("Protected Group Num: "+ protectedNum);
System.out.println("Unprotected Group Num: "+ unprotectedNum);


for (Map.Entry<String, String> entry : truthMap.entrySet()) {
	String paper = entry.getKey();
	int value = entry.getValue();
	String author = paperAuthorMap.get(paper);
	if (protectedGroup.contains(author)) { // protected Group
		if (truthMap.get(paper) == 1) 
			posProtectedNum++;
		else
			negProtectedNum++;
	} else { // Unprotected Group
		if (truthMap.get(paper) == 1)
			posUnprotectedNum++;
		else
			negUnprotectedNum++;
	}
}


/*
 * Inference
 */
println "Doing Inference"
def mpe;
if (FairnessSignal)
	mpe = new FairnessMPEInference(model, inferredDB, config,
		paperAuthorMap, protectedGroup, protectedNum, unprotectedNum, delta);
else
	mpe = new MPEInference(model, inferredDB, config);
def result = mpe.mpeInference();


inferredDB.close();

def metrics = [RankingScore.AUPRC, RankingScore.NegAUPRC, RankingScore.AreaROC, RankingScore.Kendall]

Database resultDB = data.getDatabase(inferenceWritePart, inferredPredicates);
SimpleRankingComparator comparator = new SimpleRankingComparator(resultDB);
comparator.setBaseline(wlTruthDB);
double[] score = new double[metrics.size()];
for (int r=0; r<metrics.size(); r++) {
	comparator.setRankingScore(metrics[r]);
	score[r] = comparator.compare(positiveSummary);
}

double protectedValue = 0;
double unprotectedValue = 0;

double posProtectedValue = 0;
double posUnprotectedValue = 0;
double negProtectedValue = 0;
double negUnprotectedValue = 0;

for (GroundAtom atom : Queries.getAllAtoms(resultDB, positiveSummary)) {
	GroundTerm[] terms = atom.getArguments();
	double value = atom.getValue();
	String paperName = terms[0].toString()
	String authorName = paperAuthorMap.get(paperName);
//	println authorName
	if (!protectedGroup.contains(authorName)) { // Unprotected
		unprotectedValue += value;
		if (truthMap.get(paperName))
			posUnprotectedValue += value;
		else
			negUnprotectedValue += value;
	} else { // Protected
		protectedValue += value;
		if (truthMap.get(paperName))
			posProtectedValue += value;
		else
			negProtectedValue += value;
	}
	//println ""+ value+ ", "+ truthMap.get(paperName)
}


println model.toString()
println "Area under positive PR curve: "+ score[0]
println "Area under negative PR curve: "+ score[1]
println "Area under ROC curve: "+ score[2]

double p1 = protectedValue/protectedNum;
double p2 = unprotectedValue/unprotectedNum;
println "P1: "+ p1
println "P2: "+ p2
println "RD: "+ (p1-p2)
println "RR: "+ (p1/p2)
println "RC: "+ ((1-p1)/(1-p2))

double posProtectedRate = posProtectedValue/posProtectedNum;
double posUnprotectedRate = posUnprotectedValue/posUnprotectedNum;

double negProtectedRate = negProtectedValue/negProtectedNum;
double negUnprotectedRate = negUnprotectedValue/negUnprotectedNum;

println "pos Protected Rate:"+ posProtectedRate;
println "pos Unprotected Rate: "+ posUnprotectedRate;
println "neg Protected Rate: "+ negProtectedRate
println "neg Unprotected Rate: "+ negUnprotectedRate

double pos_diff = posProtectedRate- posUnprotectedRate
double neg_diff = negProtectedRate- negUnprotectedRate

println "pos diff: "+ pos_diff
println "neg diff: "+ neg_diff


///*
// * AUC-ROC per Group
// */
//List<Double> protectedGroupValue = new ArrayList<Double>();
//List<Double> unprotectedGroupValue = new ArrayList<Double>();
//List<Integer> protectedTruthValue = new ArrayList<Integer>();
//List<Integer> unprotectedTruthValue = new ArrayList<Integer>();
//
//for (GroundAtom atom : Queries.getAllAtoms(resultDB, positiveReviews)) {
//	GroundTerm[] terms = atom.getArguments();
//	double value = atom.getValue();
//	String reviewerName = terms[0].toString();
//	String paperName = terms[1].toString();
//	String key = reviewerName+','+paperName;
//	String authorName = paperAuthorMap.get(paperName);
//	if (!protectedGroup.contains(authorName)) { // Unprotected
//		unprotectedGroupValue.add(value);
//		unprotectedTruthValue.add(posReviewTruthMap.get(key));
//	} else { // Protected
//		protectedGroupValue.add(value);
//		protectedTruthValue.add(posReviewTruthMap.get(key));
//		println ""+ value+ ", "+ truthMap.get(paperName)
//	}
////	println ""+ value+ ", "+ posReviewTruthMap.get(key)
//	
//}
//
//double[] protectedGroupArr = protectedGroupValue.toArray(); //new double[protectedGroupValue.size()];
//double[] unprotectedGroupArr = unprotectedGroupValue.toArray(); //new double[unprotectedGroupValue.size()];
//int[] protectedTruthArr = protectedTruthValue.toArray(); //new int[protectedTruthValue.size()];
//int[] unprotectedTruthArr = unprotectedTruthValue.toArray(); //new int[unprotectedTruthValue.size()];
//
//AUCcalculator calculator = new AUCcalculator();
//double protectedAUC = calculator.measure(protectedTruthArr, protectedGroupArr);
//double unprotectedAUC = calculator.measure(unprotectedTruthArr, unprotectedGroupArr)
//
//resultDB.close();
//
//
//println "Protected AUC-ROC: "+ protectedAUC
//println "Unprotcted AUC-ROC: "+ unprotectedAUC


/*
 * AUC Precision-Recall Curve Per Group 
 */
Partition writePart_protect = new Partition(555);
Partition writePart_unprotect = new Partition(556);
Partition truthPart_protect = new Partition(557);
Partition truthPart_unprotect = new Partition(558);


Inserter insert1 = data.getInserter(positiveSummary, writePart_protect);
Inserter insert2 = data.getInserter(positiveSummary, writePart_unprotect);
Inserter insert3 = data.getInserter(positiveSummary, truthPart_protect);
Inserter insert4 = data.getInserter(positiveSummary, truthPart_unprotect);

Set<GroundAtom> groundings = Queries.getAllAtoms(resultDB, positiveSummary);
for (GroundAtom ga : groundings) {
	GroundTerm[] terms = ga.getArguments();
	double value = ga.getValue();
	String paperName = terms[0].toString()
	String authorName = paperAuthorMap.get(paperName);
	if (protectedGroup.contains(authorName)) {
		insert1.insertValue(value, terms)
	} else {
		insert2.insertValue(value, terms)
	}
}
groundings.clear()
groundings = Queries.getAllAtoms(wlTruthDB, positiveSummary);
for (GroundAtom ga : groundings) {
	GroundTerm[] terms = ga.getArguments();
	double value = ga.getValue();
	String paperName = terms[0].toString();
	String authorName = paperAuthorMap.get(paperName);
	if (protectedGroup.contains(authorName)) {
		insert3.insertValue(value, terms);
	} else {
		insert4.insertValue(value, terms);
	}
}
groundings.clear();

Database resultDB_protect = data.getDatabase(writePart_protect, [positiveSummary] as Set)
Database resultDB_unprotect = data.getDatabase(writePart_unprotect, [positiveSummary] as Set)
Database truthDB_protect = data.getDatabase(truthPart_protect, [positiveSummary] as Set)
Database truthDB_unprotect = data.getDatabase(truthPart_unprotect, [positiveSummary] as Set)

// Protected Group
comparator = new SimpleRankingComparator(resultDB_protect);
comparator.setBaseline(truthDB_protect);
double[] protectScore = new double[metrics.size()];
for (int r=0; r<metrics.size(); r++) {
	comparator.setRankingScore(metrics[r]);
	protectScore[r] = comparator.compare(positiveSummary);
}
// Unprotected Group
comparator = new SimpleRankingComparator(resultDB_unprotect);
comparator.setBaseline(truthDB_unprotect);
double[] unprotectScore = new double[metrics.size()];
for (int r=0; r<metrics.size(); r++) {
	comparator.setRankingScore(metrics[r]);
	unprotectScore[r] = comparator.compare(positiveSummary);
}

println "AUPRC: "+ protectScore[0]+ ", "+ unprotectScore[0]
println "NegAUPRC: "+ protectScore[1]+ ", "+ unprotectScore[1]
println "AUC-ROC: "+ protectScore[2]+ ", "+ unprotectScore[2]



