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
import binghamton.util.DataOutputter;


// Config manager
ConfigManager cm = ConfigManager.getManager();
ConfigBundle config = cm.getBundle("paperReview-model");
Logger log = LoggerFactory.getLogger(this.class);

// Databse
def defaultPath = System.getProperty("java.io.tmpdir")
String dbpath = config.getString("dbpath", defaultPath + File.separator + "paperReview-model")
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), config) 


def fold = 'bias4';
def dataDir = 'data/biasedPaperReview/'+ fold;


boolean FairnessSignal = true;
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

Partition trainPart = new Partition(0)
Partition truthPart = new Partition(1)
Partition targetPart = new Partition(2)
Partition inferenceWritePart = new Partition(3)

def inserter;
for (Predicate p : closedPredicates) {
	println p.getName()
	String fileName = predicateFileMap[p];
	inserter = data.getInserter(p, trainPart);
	def fullFilePath = dataDir+ '/'+ fileName;
	if (predicateSoftTruthMap[p]) {
		InserterUtils.loadDelimitedDataTruth(inserter, fullFilePath, ',');
	} else {
		InserterUtils.loadDelimitedData(inserter, fullFilePath, ',');
	}
}

for (Predicate p: inferredPredicates) {
	println p.getName()
	String fileName = predicateFileMap[p];
	inserter = data.getInserter(p, truthPart);
	def fullFilePath = dataDir+ '/'+ fileName;
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

Database wlTrainDB = data.getDatabase(targetPart, closedPredicates, trainPart);
populateDatabase(data, wlTrainDB, truthPart, inferredPredicates);

Database inferredDB = data.getDatabase(inferenceWritePart, closedPredicates, trainPart);
populateDatabase(data, inferredDB, truthPart, inferredPredicates);

Database wlTruthDB = data.getDatabase(truthPart, inferredPredicates);



// Get Sensitive Attributes
def highRankInstitutes = new HashSet<>();
def instituteMap = new HashMap<>();
def studentSet = new HashSet<>();
Map<String, String> paperAuthorMap = new HashMap<>()
Set<String> protectedGroup = new HashSet<>();
//highRank
String fileName = dataDir+ '/highRank.txt';
BufferedReader reader = null;
reader = new BufferedReader(new FileReader(fileName));
String read = null;
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


/*
 * Weight Learning
 */
println "Doing Weight Learning"
def weightLearner = new MaxLikelihoodMPE(model, wlTrainDB, wlTruthDB, config);
weightLearner.learn();



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
for (GroundAtom atom : Queries.getAllAtoms(resultDB, positiveSummary)) {
	GroundTerm[] terms = atom.getArguments();
	double value = atom.getValue();
	String paperName = terms[0].toString()
	String authorName = paperAuthorMap.get(paperName);
//	println authorName
	if (!protectedGroup.contains(authorName)) { // Unprotected
		unprotectedValue += value;
	} else { // Protected
		protectedValue += value;
//		println authorName+", "+ value 
	}
}

resultDB.close();

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


/*
 * Write Weights to File
 */
def outputPath = 'result/paperReview/'+fold;
File outputDir = new File(outputPath);
if(!outputDir.exists()) {
	outputDir.mkdirs();
}

def outputFile;
if (FairnessSignal)
	outputFile = outputPath+ '/fairnessBaselineModel.txt'
else
	outputFile = outputPath+ '/baisedBaselineModel.txt'
def ps = new PrintStream(new FileOutputStream(outputFile, false));
System.setOut(ps);
for (CompatibilityKernel k : Iterables.filter(model.getKernels(), CompatibilityKernel.class)) {
	double w = k.getWeight().getWeight()
	System.out.println(""+ w)
}





