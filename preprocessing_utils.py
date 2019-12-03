import pandas as pd
import os
import re
import nltk

def make_data(root_dr,output_file):
	all_reviews_df = []
	for type_name in os.listdir(root_dr):
		all_reviews_text = []
		all_reviews_id = []
		all_reviews_pos_text = []
		all_reviews_neg_text = []
		all_reviews_label = []

		## Reading positive reviews in a directory
		path_v = root_dr + type_name + "/positive.review"
		if os.path.isfile(path_v):
			with open(path_v, 'r', encoding="ISO-8859-1") as f:
				all_lines = f.read()
				all_reviews_pos_text = re.findall("<review_text>(.*?)</review_text>", all_lines, re.DOTALL)

		## Reading negative reviews in a directory
		path_v = root_dr + type_name + "/negative.review"
		if os.path.isfile(path_v):
			with open(path_v, 'r', encoding="ISO-8859-1") as f:
				all_lines = f.read()
				all_reviews_neg_text = re.findall("<review_text>(.*?)</review_text>", all_lines, re.DOTALL)

		if len(all_reviews_pos_text):
			## Creating id's for review - I did not read the id's from the files because of pre-processing required.
			## Anyhow we don't need id's to train models.
			all_reviews_pos_id = [type_name + "_pos" + str(i) for i in range(len(all_reviews_pos_text))]
			all_reviews_neg_id = [type_name + "_neg" + str(i) for i in range(len(all_reviews_neg_text))]

			all_reviews_pos_label = [1] * len(all_reviews_pos_id)
			all_reviews_neg_label = [0] * len(all_reviews_neg_id)

			all_reviews_text.extend(all_reviews_pos_text)
			all_reviews_text.extend(all_reviews_neg_text)

			all_reviews_id.extend(all_reviews_pos_id)
			all_reviews_id.extend(all_reviews_neg_id)

			all_reviews_label.extend(all_reviews_pos_label)
			all_reviews_label.extend(all_reviews_neg_label)

			df = pd.DataFrame(data=[all_reviews_text, all_reviews_id, all_reviews_label]).T
			df.columns = ['review', 'review_id', 'label']

			all_reviews_df.append(df)
			final_df = pd.concat(all_reviews_df, sort=False)
			final_df = final_df.reset_index(drop=True)
			final_df.to_excel(output_file)

def remove_unneccessary(inputString):
	inputString = str(inputString)
	WEB_URL_REGEX = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
	inputString1 = re.sub(WEB_URL_REGEX, " __url__ ", inputString)  # removing url
	inputString2 = re.sub(r"(?:@[\w_]+)", " __mention__ ", inputString1)
	inputString3 = re.sub(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", " __hashtag__ ", inputString2)
	inputString4 = re.sub("\s+", " ", inputString3)  ## Multiple spaces
	inputString5 = re.sub(r"[^\s]+@([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?", " __email__ ",
						  inputString4)  # replace email with _email_
	inputString6 = re.sub("\d+", " __number__ ", inputString5)  # remove numbers

	inputString7 = re.sub(r'(.)\1+', r'\1\1', inputString6)  # multiple occurrences of same word

	return inputString7

def preprocess_data(sentences_series):
	sentences_series = sentences_series.apply(lambda x: re.sub("\n", "", x))
	fix_spaces_puncs = re.compile(r'\s*([?!.,]+(?:\s+[?!.,]+)*)\s*')
	sentences_series = sentences_series.apply(
		lambda x: fix_spaces_puncs.sub(lambda y: "{} ".format(y.group(1).replace(" ","")), x))

	sentences_series = sentences_series.apply(remove_unneccessary)
	sentences_series = sentences_series.apply(lambda x: x.lower())
	return sentences_series

def tokenise_and_rem_stop(sentence_series,augmented_words=None):
	nltk_words = nltk.corpus.stopwords.words('english')
	final_stops = []
	## removing words containing n't
	for i in nltk_words:
		if "n't" not in i:
			final_stops.append(i)

	if augmented_words:
		final_stops.append(augmented_words)

	tokens_samples = sentence_series.apply(lambda x : nltk.word_tokenize(x))
	tokens_samples = tokens_samples.apply(lambda tokens: [i for i in tokens if i not in final_stops])
	return tokens_samples

