/*Using names.txt (right click and 'Save Link/Target As...'), a 46K text file containing over five-thousand first names, begin by sorting it into alphabetical order. Then working out the alphabetical value for each name, multiply this value by its alphabetical position in the list to obtain a name score.

For example, when the list is sorted into alphabetical order, COLIN, which is worth 3 + 15 + 12 + 9 + 14 = 53, is the 938th name in the list. So, COLIN would obtain a score of 938 × 53 = 49714.

What is the total of all the name scores in the file?
*/

#include<iostream>
#include<string>
#include<cstring>
#include<fstream>
#include<vector>
#include<sstream>
#include<algorithm>
#include<boost/algorithm/string.hpp>
using namespace std;

int getNameScore(string& name){

	string alphabet = "abcdefghijklmnopqrstuvwxyz";
	boost::to_upper(alphabet);
	int sizeOfChar = sizeof(name[0]);
	int nLetters = sizeof(name)/sizeOfChar;
	int letter = 0;
	int score = 0;
	for(letter = 0; letter<= nLetters; ++letter){

		string::size_type position = alphabet.find(name[letter]);	
		//cout << position << endl;
		score += position +1 ;
	}
	//cout << "Score for name " << name << " = " << score << "\n";
	//cout << "nletters " << nLetters << endl;
	//cout << "sizeof char " << sizeOfChar<< endl;
	return score;	
}

int main (){


	long long totalScore = 0;
	string line;
	//string filePath = "~/c/p022_names.txt";
	ifstream file("/home/msmith/c/p022_names.txt");
	if (file.is_open())
	{
		vector<string> words;
		string name;
		while(getline(file,name,',')){
			words.push_back(name);
			//cout << name << endl;
		}
		file.close();
		sort(words.begin(), words.end());
		//printVec(words);		
		//cout << "Number of names = " << words.size() << '\n';
		long long count = 1;

		for(std::vector<string>::iterator it = words.begin(); it != words.end(); ++it){
			int nameScore = getNameScore(*it);
			totalScore += count*nameScore;
			cout << count << " Name " << *it << " .Count = "<<count<< endl;
			cout << "Name score = " << nameScore<< endl;
			cout << "Score = " << nameScore*count<< endl;
			count += 1;
		}

	

	}else{
		cout << "Couldn't open "<< endl;
		return -1;
	}

	cout << "Total = " << totalScore << endl;

	return 0;
}

