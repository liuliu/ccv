#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <bitset>

static int _CCV_PRINT_COUNT __attribute__ ((unused)) = 0;
static int _CCV_PRINT_LOOP __attribute__ ((unused)) = 0;

#define FLUSH(a, ...) \
	do { \
		for (_CCV_PRINT_LOOP = 0; _CCV_PRINT_LOOP < _CCV_PRINT_COUNT; _CCV_PRINT_LOOP++) \
			printf("\b"); \
		for (_CCV_PRINT_LOOP = 0; _CCV_PRINT_LOOP < _CCV_PRINT_COUNT; _CCV_PRINT_LOOP++) \
			printf(" "); \
		for (_CCV_PRINT_LOOP = 0; _CCV_PRINT_LOOP < _CCV_PRINT_COUNT; _CCV_PRINT_LOOP++) \
			printf("\b"); \
		_CCV_PRINT_COUNT = printf(a, ##__VA_ARGS__); \
		fflush(stdout); \
	} while (0) // using do while (0) to force ; line end

using namespace std;

typedef struct ccv_darts_tree {
	map<string, struct ccv_darts_tree*> super;
	map<string, struct ccv_darts_tree*> child;
	bitset<1000> leaf_des;
	string synset;
	double info_gain;
	double probs;
	double correct;
	int inverse_high;
} ccv_darts_tree_t;

#define PI (3.141592653589793)

static double erfi(double x)
{
	double a = 0.147;
	double a1 = log(1.0 - x * x);
	double a2 = 2.0 / (PI * a) + a1 / 2.0;
	return (x > 0 ? 1 : -1) * sqrt(sqrt(a2 * a2 - a1 / a) - a2);
}

static double normcdfi(double x)
{
	return sqrt(2.0) * erfi(2 * x - 1);
}

static double binofit(double p, double N, double c)
{
	double z = normcdfi(1.0 - 0.5 * (1.0 - c));
	double a1 = 1.0 / (1.0 + z * z / N);
	double a2 = p + z * z / (2 * N);
	double a3 = z * sqrt(p * (1 - p) / N + z * z / (4 * N * N));

	return a1 * (a2 - a3); // only return lower bound, upper bound is: a1 * (a2 + a3)
}

static void recompute_leaf_des(ccv_darts_tree_t* node)
{
	int i;
	for (map<string, ccv_darts_tree_t*>::iterator it = node->child.begin(); it != node->child.end(); ++it)
		for (i = 0; i < 1000; i++)
			node->leaf_des[i] = (node->leaf_des[i] || it->second->leaf_des[i]);
	for (map<string, ccv_darts_tree_t*>::iterator it = node->super.begin(); it != node->super.end(); ++it)
		recompute_leaf_des(it->second);
}

static void recompute_probs(ccv_darts_tree_t* node)
{
	double probs = 0;
	for (map<string, ccv_darts_tree_t*>::iterator it = node->child.begin(); it != node->child.end(); ++it)
		probs += it->second->probs / it->second->super.size();
	assert(probs <= 1 + 1e-5);
	node->probs = probs < 1 ? probs : 1;
}

static void recompute_inverse_high(ccv_darts_tree_t* node)
{
	int inverse_high = 0;
	for (map<string, ccv_darts_tree_t*>::iterator it = node->child.begin(); it != node->child.end(); ++it)
		inverse_high = min(inverse_high, it->second->inverse_high);
	node->inverse_high = inverse_high - 1;
	for (map<string, ccv_darts_tree_t*>::iterator it = node->super.begin(); it != node->super.end(); ++it)
		recompute_inverse_high(it->second);
}

static void recompute_correctness(double correct, ccv_darts_tree_t* node)
{
	node->correct = correct;
	for (map<string, ccv_darts_tree_t*>::iterator it = node->super.begin(); it != node->super.end(); ++it)
		recompute_correctness(correct, it->second);
}

int main(int argc, char** argv)
{
	ifstream ifwnid;
	ifwnid.open(argv[1]);
	string wnids[1000];
	int i, j, k;
	for (i = 0; i < 1000; i++)
		ifwnid >> wnids[i];
	ifwnid.close();
	ifstream iftree;
	iftree.open(argv[2]);
	map<string, ccv_darts_tree_t*> tree_map;
	while (!iftree.eof())
	{
		string synset_a, synset_b;
		iftree >> synset_a >> synset_b;
		ccv_darts_tree_t* node_a;
		if (tree_map.count(synset_a) > 0)
			node_a = tree_map[synset_a];
		else {
			node_a = new ccv_darts_tree_t;
			node_a->leaf_des = 0;
			node_a->inverse_high = 0;
			node_a->synset = synset_a;
			tree_map[synset_a] = node_a;
		}
		ccv_darts_tree_t* node_b;
		if (tree_map.count(synset_b) > 0)
			node_b = tree_map[synset_b];
		else {
			node_b = new ccv_darts_tree_t;
			node_b->leaf_des = 0;
			node_b->inverse_high = 0;
			node_b->synset = synset_b;
			tree_map[synset_b] = node_b;
		}
		node_a->child[synset_b] = node_b;
		node_b->super[synset_a] = node_a;
	}
	iftree.close();
	// Compute information gain for each node
	for (i = 0; i < 1000; i++)
	{
		ccv_darts_tree_t* node = tree_map[wnids[i]];
		node->leaf_des[i] = 1;
		for (map<string, ccv_darts_tree_t*>::iterator it = node->super.begin(); it != node->super.end(); ++it)
			recompute_leaf_des(it->second);
	}
	map<string, ccv_darts_tree_t*> used;
	double min_rewards = 0;
	double max_rewards = log(1000.0 / 1.0) / log(2.);
	for (map<string, ccv_darts_tree_t*>::iterator it = tree_map.begin(); it != tree_map.end(); ++it)
		if (it->second->leaf_des.count() > 0)
		{
			it->second->info_gain = log(1000.0 / it->second->leaf_des.count()) / log(2.);
			used[it->first] = it->second;
		}
	ifstream ifval;
	ifval.open(argv[3]);
	double* probs = (double*)calloc(50000 * 1000, sizeof(double));
	for (i = 0; i < 50000; i++)
	{
		for (j = 0; j < 1000; j++)
		{
			int idx;
			double val;
			ifval >> idx >> val;
			// idx from 1~1000 to 0~999
			probs[i * 1000 + idx - 1] = val;
		}
		double val = 0;
		for (j = 0; j < 1000; j++)
			val += probs[i * 1000 + j];
		// compensate for some accuracy loss, re-normalize
		val = 1.0 / val;
		for (j = 0; j < 1000; j++)
			probs[i * 1000 + j] = probs[i * 1000 + j] * val;
	}
	ifval.close();
	ifstream iftruth;
	iftruth.open(argv[4]);
	int* truth = (int*)calloc(50000, sizeof(int));
	for (i = 0; i < 50000; i++)
	{
		iftruth >> truth[i];
		truth[i] = truth[i] - 1; // 1~1000 to 0~999
	}
	iftruth.close();
	const double accuracy_guarantee = 0.99;
	const double epsilon = 1 - accuracy_guarantee;
	const double confidence = 0.95;
	double min_lambda = 0;
	double max_lambda = ((1 - epsilon) * max_rewards - min_rewards) / epsilon;
	printf("Binary search between %lf and %lf\n", min_lambda, max_lambda);
	// Top-sort for used
	for (i = 0; i < 1000; i++)
	{
		ccv_darts_tree_t* node = used[wnids[i]];
		node->inverse_high = 0;
		for (map<string, ccv_darts_tree_t*>::iterator it = node->super.begin(); it != node->super.end(); ++it)
			recompute_inverse_high(it->second);
	}
	vector<ccv_darts_tree_t*> sort;
	for (i = 0; i > -(int)used.size(); i--) // Arbitrary large negative numbers
	{
		int found = 0;
		for (map<string, ccv_darts_tree_t*>::iterator it = used.begin(); it != used.end(); ++it)
			if (it->second->inverse_high == i)
				sort.push_back(it->second), found = 1;
		if (!found)
			break;
	}
	int max_high = abs(i);
	double* accuracy_at_high = (double*)calloc(max_high, sizeof(double));
	assert(used.size() == sort.size());
	for (i = 0; i < 25; i++)
	{
		double current_lambda = (min_lambda + max_lambda) / 2.0;
		double correct = 0;
		for (j = 0; j < max_high; j++)
			accuracy_at_high[j] = 0;
		for (j = 0; j < 50000; j++)
		{
			if (j % 291 == 0 || j == 49999)
				FLUSH("At %d / %d, going over %d / %d", i + 1, 25, j + 1, 50000);
			for (map<string, ccv_darts_tree_t*>::iterator it = used.begin(); it != used.end(); ++it)
				it->second->probs = 0;
			for (k = 0; k < 1000; k++)
			{
				ccv_darts_tree_t* node = used[wnids[k]];
				node->probs = probs[j * 1000 + k];
			}
			for (vector<ccv_darts_tree_t*>::iterator it = sort.begin(); it != sort.end(); ++it)
			{
				if ((*it)->inverse_high < 0)
					recompute_probs(*it);
				(*it)->correct = 0;
			}
			ccv_darts_tree_t* truth_node = used[wnids[truth[j]]];
			recompute_correctness(1, truth_node);
			string max_wnid = "";
			double max_rewards = 0;
			for (map<string, ccv_darts_tree_t*>::iterator it = used.begin(); it != used.end(); ++it)
			{
				double rewards = (it->second->info_gain + current_lambda) * it->second->probs;
				if (rewards > max_rewards)
				{
					max_wnid = it->first;
					max_rewards = rewards;
				}
			}
			assert(max_wnid.size() > 0);
			correct += used[max_wnid]->correct;
			accuracy_at_high[abs(used[max_wnid]->inverse_high)] += 1;
		}
		double accuracy = correct / 50000.0;
		double accuracy_lower_bound = binofit(accuracy, 50000, confidence);
		if (accuracy_lower_bound > accuracy_guarantee)
			max_lambda = current_lambda;
		else
			min_lambda = current_lambda;
		printf("\nAt %d / %d, lambda %lf, at accuracy %.3lf%%, accuracy lower bound %.3lf%%\n", i + 1, 25, current_lambda, accuracy * 100, accuracy_lower_bound * 100);
		printf("accuracy at: (%d, %.3lf%%)", 0, accuracy_at_high[0] * 100 / 50000.0);
		for (j = 1; j < max_high; j++)
			printf(", (%d, %.3lf%%)", j, accuracy_at_high[j] * 100 / 50000.0);
		printf("\n");
	}
	free(probs);
	return 0;
}
