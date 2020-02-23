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
model.add predicate: "affiliate2HighRank", types: [ArgumentType.UniqueID]

def closedPredicates = [author, paper, reviewer, student, acceptable, 
	reviews, submits, institute, highRank, affiliation, affiliate2HighRank] as Set;
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
	((Predicate)affiliation):"affiliation.txt",
	((Predicate)affiliate2HighRank):"affiliate2HighRank.txt"]


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
	((Predicate)affiliation):false,
	((Predicate)affiliate2HighRank):true]


def initWeight = 5.0

/*
model.add(rule:  ( STUDENT(A) & SUBMITS(A, P) ) >> ~( POSITIVESUMMARY(P) ) , weight: 4.645570, sqaured:true)
model.add(rule:  ( ( ( ( ( AUTHOR(A) & REVIEWS(R2, P) ) & SUBMITS(A, P) ) & POSITIVEREVIEWS(R, P) ) & POSITIVESUMMARY(P) ) & ~( POSITIVEREVIEWS(R2, P) ) ) >> AFFILIATE2HIGHRANK(A) , weight: 4.675310, sqaured:true)
model.add(rule:  ( ( ( ( ( AUTHOR(A) & PAPER(P) ) & REVIEWS(R, P) ) & ACCEPTABLE(P) ) & ~( AFFILIATE2HIGHRANK(A) ) ) & ~( POSITIVEREVIEWS(R, P) ) ) >> POSITIVESUMMARY(P) , weight: 4.517209, sqaured:true)
model.add(rule:  ( ( ( ( AUTHOR(A) & REVIEWS(R2, P) ) & SUBMITS(A, P) ) & POSITIVEREVIEWS(R, P) ) & ~( STUDENT(A) ) ) >> POSITIVEREVIEWS(R2, P) , weight: 4.419188, sqaured:true)
model.add(rule:  ( ( ( ( ( PAPER(P) & STUDENT(A) ) & REVIEWS(R2, P) ) & SUBMITS(A, P) ) & POSITIVEREVIEWS(R, P) ) & POSITIVESUMMARY(P) ) >> ACCEPTABLE(P) , weight: 4.892884, sqaured:true)
model.add(rule:  ( ( ( ( AUTHOR(A) & PAPER(P) ) & POSITIVEREVIEWS(R, P) ) & POSITIVESUMMARY(P) ) & ~( ACCEPTABLE(P) ) ) >> AFFILIATE2HIGHRANK(A) , weight: 4.921369, sqaured:true)
model.add(rule:  ( ( ( ( ( AUTHOR(A) & REVIEWS(R2, P) ) & AFFILIATE2HIGHRANK(A) ) & POSITIVESUMMARY(P) ) & ~( STUDENT(A) ) ) & ~( POSITIVEREVIEWS(R2, P) ) ) >> ~( POSITIVEREVIEWS(R, P) ) , weight: 4.948252, sqaured:true)
model.add(rule:  ( ( ( PAPER(P) & REVIEWS(R2, P) ) & POSITIVEREVIEWS(R, P) ) & ~( POSITIVESUMMARY(P) ) ) >> ~( SUBMITS(A, P) ) , weight: 5.187841, sqaured:true)
model.add(rule:  ( ( ( ( AUTHOR(A) & REVIEWS(R2, P) ) & ~( STUDENT(A) ) ) & ~( AFFILIATE2HIGHRANK(A) ) ) & ~( POSITIVEREVIEWS(R2, P) ) ) >> ~( POSITIVEREVIEWS(R, P) ) , weight: 4.629614, sqaured:true)
model.add(rule:  ( ( ( ( ( AUTHOR(A) & PAPER(P) ) & REVIEWS(R2, P) ) & ~( STUDENT(A) ) ) & ~( ACCEPTABLE(P) ) ) & ~( POSITIVEREVIEWS(R2, P) ) ) >> ~( AFFILIATE2HIGHRANK(A) ) , weight: 2.567480, sqaured:true)
*/

/*
// Fairness-A3SL Model
model.add(rule:  ( ( SUBMITS(A, P) & POSITIVEREVIEWS(R2, P) ) & POSITIVESUMMARY(P) ) >> ~( STUDENT(A) ) , weight: 4.754780, sqaured:true)
model.add(rule:  ( ( ( ( PAPER(P) & ACCEPTABLE(P) ) & REVIEWS(R, P) ) & SUBMITS(A, P) ) & POSITIVEREVIEWS(R, P) ) >> POSITIVESUMMARY(P) , weight: 4.707828, sqaured:true)
model.add(rule:  ( ( ( ( REVIEWS(R, P) & STUDENT(A) ) & ACCEPTABLE(P) ) & POSITIVEREVIEWS(R2, P) ) & ~( POSITIVEREVIEWS(R, P) ) ) >> ~( REVIEWS(R2, P) ) , weight: 4.595191, sqaured:true)
model.add(rule:  ( ( ( REVIEWS(R, P) & REVIEWS(R2, P) ) & STUDENT(A) ) & ~( POSITIVEREVIEWS(R, P) ) ) >> POSITIVEREVIEWS(R2, P) , weight: 0.531194, sqaured:true)
model.add(rule:  ( ( ( REVIEWS(R2, P) & REVIEWS(R, P) ) & POSITIVESUMMARY(P) ) & ~( POSITIVEREVIEWS(R2, P) ) ) >> ~( POSITIVEREVIEWS(R, P) ) , weight: 4.954562, sqaured:true)
model.add(rule:  ( ( PAPER(P) & SUBMITS(A, P) ) & ~( ACCEPTABLE(P) ) ) >> ~( POSITIVEREVIEWS(R2, P) ) , weight: 5.279860, sqaured:true)
model.add(rule:  ( ( REVIEWS(R2, P) & STUDENT(A) ) & ~( POSITIVEREVIEWS(R2, P) ) ) >> ~( ACCEPTABLE(P) ) , weight: 4.460578, sqaured:true)
*/

/*
// 0.5
model.add(rule:  ( ( ( PAPER(P) & STUDENT(A) ) & REVIEWS(R, P) ) & ~( POSITIVESUMMARY(P) ) ) >> POSITIVEREVIEWS(R, P) , weight: 0.490039, sqaured:true)
model.add(rule:  ( ( ( REVIEWS(R, P) & STUDENT(A) ) & POSITIVESUMMARY(P) ) & ~( POSITIVEREVIEWS(R, P) ) ) >> ~( SUBMITS(A, P) ) , weight: 4.940628, sqaured:true)
model.add(rule:  ( ( ( ( ( PAPER(P) & REVIEWS(R, P) ) & ACCEPTABLE(P) ) & POSITIVEREVIEWS(R2, P) ) & ~( POSITIVEREVIEWS(R, P) ) ) & ~( POSITIVESUMMARY(P) ) ) >> ~( SUBMITS(A, P) ) , weight: 4.662659, sqaured:true)
model.add(rule:  ( ( ( ( PAPER(P) & REVIEWS(R2, P) ) & SUBMITS(A, P) ) & POSITIVEREVIEWS(R, P) ) & ~( POSITIVESUMMARY(P) ) ) >> ~( ACCEPTABLE(P) ) , weight: 4.257851, sqaured:true)
model.add(rule:  ( ( REVIEWS(R2, P) & POSITIVEREVIEWS(R, P) ) & ~( POSITIVEREVIEWS(R2, P) ) ) >> ~( POSITIVESUMMARY(P) ) , weight: 4.954562, sqaured:true)
model.add(rule:  ( ( ( PAPER(P) & REVIEWS(R, P) ) & REVIEWS(R2, P) ) & SUBMITS(A, P) ) >> POSITIVESUMMARY(P) , weight: 2.516072, sqaured:true)
model.add(rule:  ( ( ( PAPER(P) & STUDENT(A) ) & ACCEPTABLE(P) ) & REVIEWS(R2, P) ) >> POSITIVESUMMARY(P) , weight: 3.785574, sqaured:true)
model.add(rule:  ( ( PAPER(P) & STUDENT(A) ) & POSITIVEREVIEWS(R, P) ) >> POSITIVESUMMARY(P) , weight: 4.591061, sqaured:true)
*/


// 0.1
model.add(rule:  ( ( ( ( AUTHOR(A) & REVIEWS(R2, P) ) & SUBMITS(A, P) ) & POSITIVEREVIEWS(R, P) ) & ~( STUDENT(A) ) ) >> POSITIVEREVIEWS(R2, P) , weight: 4.741861, sqaured:true)
model.add(rule:  ( ( REVIEWS(R, P) & POSITIVESUMMARY(P) ) & ~( POSITIVEREVIEWS(R, P) ) ) >> ~( SUBMITS(A, P) ) , weight: 4.954562, sqaured:true)
model.add(rule:  ( ( ( ( PAPER(P) & REVIEWS(R2, P) ) & ACCEPTABLE(P) ) & ~( POSITIVEREVIEWS(R2, P) ) ) & ~( POSITIVESUMMARY(P) ) ) >> ~( REVIEWS(R, P) ) , weight: 4.527723, sqaured:true)
model.add(rule:  ( ( ( ( AUTHOR(A) & PAPER(P) ) & REVIEWS(R2, P) ) & ~( STUDENT(A) ) ) & ~( POSITIVEREVIEWS(R2, P) ) ) >> POSITIVESUMMARY(P) , weight: 2.925012, sqaured:true)
model.add(rule:  ( ( REVIEWS(R, P) & ACCEPTABLE(P) ) & ~( POSITIVEREVIEWS(R, P) ) ) >> ~( POSITIVESUMMARY(P) ) , weight: 4.932532, sqaured:true)
model.add(rule:  ( ( ( ( ( ( AUTHOR(A) & PAPER(P) ) & REVIEWS(R2, P) ) & ACCEPTABLE(P) ) & POSITIVEREVIEWS(R, P) ) & ~( STUDENT(A) ) ) & ~( POSITIVEREVIEWS(R2, P) ) ) >> POSITIVESUMMARY(P) , weight: 4.662659, sqaured:true)


/*
model.add(rule:  ( ( ( ( PAPER(P) & REVIEWS(R2, P) ) & STUDENT(A) ) & REVIEWS(R, P) ) & ~( POSITIVEREVIEWS(R2, P) ) ) >> POSITIVESUMMARY(P) , weight: 0.490039, sqaured:true)
model.add(rule:  ( ( ( ( ( AUTHOR(A) & PAPER(P) ) & REVIEWS(R, P) ) & SUBMITS(A, P) ) & ~( STUDENT(A) ) ) & ~( POSITIVEREVIEWS(R, P) ) ) >> POSITIVESUMMARY(P) , weight: 3.193028, sqaured:true)
model.add(rule:  ( ( AUTHOR(A) & PAPER(P) ) & ~( STUDENT(A) ) ) >> POSITIVESUMMARY(P) , weight: 2.537304, sqaured:true)
model.add(rule:  ( ( ( PAPER(P) & SUBMITS(A, P) ) & POSITIVEREVIEWS(R, P) ) & ~( ACCEPTABLE(P) ) ) >> ~( POSITIVEREVIEWS(R2, P) ) , weight: 4.789027, sqaured:true)
model.add(rule:  ( ( SUBMITS(A, P) & POSITIVEREVIEWS(R2, P) ) & POSITIVESUMMARY(P) ) >> ~( REVIEWS(R2, P) ) , weight: 2.178369, sqaured:true)
model.add(rule:  ( ( ( ( ( AUTHOR(A) & PAPER(P) ) & ACCEPTABLE(P) ) & POSITIVEREVIEWS(R2, P) ) & ~( STUDENT(A) ) ) & ~( POSITIVESUMMARY(P) ) ) >> ~( POSITIVEREVIEWS(R, P) ) , weight: 4.426521, sqaured:true)
model.add(rule:  ( ( ( ( PAPER(P) & REVIEWS(R, P) ) & ~( ACCEPTABLE(P) ) ) & ~( POSITIVEREVIEWS(R, P) ) ) & ~( POSITIVESUMMARY(P) ) ) >> ~( SUBMITS(A, P) ) , weight: 0.380135, sqaured:true)
model.add(rule:  ( REVIEWS(R, P) & POSITIVESUMMARY(P) ) >> POSITIVEREVIEWS(R, P) , weight: 5.411794, sqaured:true)
*/



Partition trainPart = new Partition(0)
Partition truthPart = new Partition(1)
//Partition targetPart = new Partition(2)
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

//Database inferredDB = data.getDatabase(inferenceWritePart, closedPredicates, trainPart);
//populateDatabase(data, inferredDB, truthPart, inferredPredicates);

Database wlTruthDB = data.getDatabase(truthPart, inferredPredicates);

Database inferredDB = data.getDatabase(inferenceWritePart, closedPredicates, trainPart);

for (int j=0; j<inferredPredicates.size(); j++) {
	ResultList allGroundings = wlTruthDB.executeQuery(Queries.getQueryForAllAtoms(inferredPredicates[j]));
	for (int i=0; i<allGroundings.size(); i++) {
		GroundTerm [] grounding = allGroundings.get(i);
		GroundAtom atom = inferredDB.getAtom(inferredPredicates[j], grounding);
		if (atom instanceof RandomVariableAtom) {
			inferredDB.commit((RandomVariableAtom) atom);
		}
	}
}

Map<String, Integer> truthMap = new HashMap<>();
String fileName = dataDir+ '/positiveSummary.txt'
BufferedReader reader = new BufferedReader(new FileReader(fileName));
String read = null;
while((read = reader.readLine()) != null) {
	String[] splited = read.split(",");
	int value = Integer.parseInt(splited[1]);
	truthMap.put(splited[0], value);
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
// student
fileName = dataDir+ '/student.txt';
reader = new BufferedReader(new FileReader(fileName));
read = null;
while((read = reader.readLine()) != null) {
	String[] splited = read.split(',');
	String s = splited[0];
	int value = Integer.parseInt(splited[1]);
	if (value == 1)
		studentSet.add(s);
}
// submits
fileName = dataDir+ '/submits.txt';
reader = new BufferedReader(new FileReader(fileName));
read = null;
while((read = reader.readLine()) != null) {
	String[] splited = read.split(',');
	String a = splited[0];
	String p = splited[1];
	paperAuthorMap.put(p, a);
} 

int protectedNum = 0;
int unprotectedNum = 0;

int posProtectedNum = 0;
int negProtectedNum = 0;
int posUnprotectedNum = 0;
int negUnprotectedNum = 0;

//for (Map.Entry<String, String> entry : instituteMap.entrySet()) {
//	String stud = entry.getKey();
//	String inst = entry.getValue();
//	if (studentSet.contains(stud)) {
//		if (highRankInstitutes.contains(inst)) {
//			unprotectedNum++; // Unprotected Group
//		}
//		else {
//			protectedNum++; // Protected Group
//			protectedGroup.add(stud);
//		}
//	} else {
//		unprotectedNum++;
//	}
//}

for (Map.Entry<String, Integer> entry : truthMap.entrySet()) {
	String paper = entry.getKey();
	String author = paperAuthorMap.get(paper);
	String institute = instituteMap.get(author);
	if (highRankInstitutes.contains(institute) || !studentSet.contains(author)) { // unprotected group
		unprotectedNum++;
		if (truthMap.get(paper)==1) {
			posUnprotectedNum++;
		} else {
			negUnprotectedNum++;
		}
	} else { // protected group
		protectedNum++;
		protectedGroup.add(author);
		if (truthMap.get(paper)==1) {
			posProtectedNum++;
		} else {
			negProtectedNum++;
		}
	}
}

System.out.println("Protected Group Num: "+ protectedNum);
System.out.println("Unprotected Group Num: "+ unprotectedNum);

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



//// Debug
//for (GroundAtom atom : Queries.getAllAtoms(resultDB, positiveSummary)) {
//	GroundTerm[] terms = atom.getArguments();
//	double value = atom.getValue();
//	System.out.println(terms[0].toString()+ ": "+ value);
//}


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
 * Equal Odds
 */
double posProtectedValue = 0;
double posUnprotectedValue = 0;
double negProtectedValue = 0;
double negUnprotectedValue = 0;

for (GroundAtom atom : Queries.getAllAtoms(resultDB, positiveSummary)) {
	GroundTerm[] terms = atom.getArguments();
	double value = atom.getValue();
	String paperName = terms[0].toString()
	String authorName = paperAuthorMap.get(paperName);
	if (!protectedGroup.contains(authorName)) { // Unprotected
		if (truthMap.get(paperName) == 1) {
			posUnprotectedValue += value;
		} else {
			negUnprotectedValue += value;
		}
	} else { // Protected
		if (truthMap.get(paperName) == 1) {
			posProtectedValue += value;
		} else {
			negProtectedValue += value;
		}
	}
}


//println ""+ (posUnprotectedValue+ negUnprotectedValue)+ ", "+ unprotectedValue

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
//for (GroundAtom atom : Queries.getAllAtoms(resultDB, positiveSummary)) {
//	GroundTerm[] terms = atom.getArguments();
//	double value = atom.getValue();
//	String paperName = terms[0].toString()
//	String authorName = paperAuthorMap.get(paperName);
//	if (!protectedGroup.contains(authorName)) { // Unprotected
//		unprotectedGroupValue.add(value);
//		unprotectedTruthValue.add(truthMap.get(paperName));
//	} else { // Protected
//		protectedGroupValue.add(value);
//		protectedTruthValue.add(truthMap.get(paperName));
//	}
//	println paperName+ ", "+ value+ ", "+ truthMap.get(paperName)
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








