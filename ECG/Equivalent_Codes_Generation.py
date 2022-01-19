import operator
import pandas as pd
import os
import json
from sklearn.metrics.pairwise import cosine_similarity

NUMBER_PARENT = 15  # Number of most important parent column
NUMBER_SIMILARITY_PARENT = 200
features_to_drop = ['parent']  # Feature we don't change them
break_after_one_example = True
dic_important = {}
#  region changeable
dict_class_to_feature = {"delta_IfStatement": ["ChangeIfToFor", "ChangeIfToWhile"],
                         "current_IfStatement": ["ChangeIfToFor", "ChangeIfToWhile"],
                         "ast_diff_condBranIfAdd": ["ChangeIfToFor", "ChangeIfToWhile"],
                         "ast_diff_condExpExpand": ["ChangeIfToFor", "ChangeIfToWhile"],
                         "Remove_row": ["Unreachable"],
                         "Unreachable": ["Unreachable"],
                         "ast_diff_mcParAdd": ["ChangeMcParAdd"],
                         "ast_diff_mcRem": ["mcRem"],
                         "ast_diff_loopInitChange": ["ChangeloopInitChange"],
                         }

changeable = list(dict_class_to_feature.keys())


def read_importance_file(name_project, name_model=None):
    import Algorithms.RF as RF
    global dic_important
    path = f"{RF.name_models}/{RF.features_name}/importances.json"
    if name_model is not None:
        path = f"{name_model}/importances_{name_model}.json"
    print(path)
    if not os.path.exists(
            os.path.join("Data/" + name_project, "Results", path)):
        name_project = "../Data/" + name_project
    with open(os.path.join(name_project, "Results", path),
              'r') as f:
        dic_important = json.load(f)[0]
        features_list = dic_important.keys()
    dic_important = sorted(dic_important.items(), key=operator.itemgetter(1), reverse=True)
    return dic_important, features_list


def find_k_parent(negative_classify_sample, tp_sample, name_project, name_model):
    global dic_important
    dic_important, features_list = read_importance_file(name_project, name_model)
    most_parent_important = []
    for key, value in dic_important:
        if len(most_parent_important) == NUMBER_PARENT: break
        if key.startswith('parent'): most_parent_important.append(key)
    dic_important = dict(dic_important)
    negative_classify_sample_ = negative_classify_sample.drop(set(features_list) - set(most_parent_important), axis=1)
    tp_sample_ = tp_sample.drop(set(features_list) - set(most_parent_important))
    similarity = cosine_similarity([tp_sample_.values], negative_classify_sample_.values).tolist()[0]
    n_sample = pd.DataFrame(columns=negative_classify_sample.columns)
    for ind, sim in sorted(list(enumerate(similarity)), key=lambda x: x[1], reverse=True):
        if len(n_sample) == NUMBER_SIMILARITY_PARENT: break
        n_sample.loc[len(n_sample)] = negative_classify_sample.iloc[ind]
    return n_sample


def remove_parent_col(negative_classify_sample_, tp_sample_):
    negative_classify_sample_ = negative_classify_sample_.drop(
        columns=list(filter(lambda c: any(map(lambda f: f in c, features_to_drop)), negative_classify_sample_.columns)),
        axis=1)
    tp_sample_ = tp_sample_.drop(
        list(filter(lambda c: any(map(lambda f: f in c, features_to_drop)), list(tp_sample_.axes[0]))))
    return tp_sample_, negative_classify_sample_


def remove_row(tp, n):  # because is used we can remove comment
    remove_tp = (tp['used_added_lines+used_removed_lines'] - tp['used_added_lines-used_removed_lines']) // 2
    remove_n = (n['used_added_lines+used_removed_lines'] - n['used_added_lines-used_removed_lines']) // 2
    if remove_tp > remove_n:
        return "Remove_row", remove_tp - remove_n
    return None


def update_dict_unchanged(key_changeable):
    dict_unchanged = json.load(open("../../Play/dict_unchanged.json"))
    val = dict_unchanged.get(key_changeable, 0)
    dict_unchanged[key_changeable] = val + 1
    json.dump(dict_unchanged, open("../../Play/dict_unchanged.json", 'w'))


def update_key_rules(key_add):
    dict_rules = json.load(open("../../Play/key_rules.json"))
    for key in key_add:
        val = dict_rules.get(key, 0)
        dict_rules[key] = val + 1
    json.dump(dict_rules, open("../../Play/key_rules.json", 'w'))


def do_changeable(key_feature, key_move, x, number_apply):
    classes = dict_class_to_feature[key_move]
    change_X = x
    for class_check in classes:
        change_X = x
        instance = eval(class_check)()
        copy_x = x.copy()
        can_change = True
        for i in range(int(number_apply)):
            if not instance.apply_if_can(copy_x, key_feature):
                can_change = False
                break
        if can_change:
            change_X = copy_x
            break
    return change_X


def check_changeable(n_samples, tp_dic):
    can_change = True
    X_changeable = tp_dic.copy()
    for key_changeable in n_samples:
        if key_changeable.startswith("parent") or X_changeable[key_changeable] == n_samples[key_changeable]:
            continue
        # sample that we can't change
        elif key_changeable not in changeable and X_changeable[key_changeable] > n_samples[key_changeable]:
            continue
        elif key_changeable in changeable and X_changeable[key_changeable] > n_samples[key_changeable]:
            # apply changeable
            X_changeable = do_changeable(key_changeable, key_changeable, X_changeable,
                                         n_samples[key_changeable] - X_changeable[key_changeable])

            if X_changeable == tp_dic:
                if dic_important[key_changeable] > 0.001:
                    can_change = False
                    break
        elif X_changeable[key_changeable] < n_samples[key_changeable]:
            X_changeable = do_changeable(key_changeable, "Unreachable", X_changeable,
                                         n_samples[key_changeable] - X_changeable[key_changeable])
            if X_changeable == tp_dic:
                if dic_important[key_changeable] > 0.001:
                    can_change = False
                    break
    return can_change, X_changeable


def get_similarities(tp_sample, negative_samples, name_project, name_model):
    negative_classify_sample_ = find_k_parent(negative_samples, tp_sample, name_project, name_model)
    tp_sample_, negative_classify_sample_ = remove_parent_col(negative_classify_sample_, tp_sample)
    tp_dic = tp_sample.to_dict()
    # check cosine similarity with negative and tp (without parent column)
    similarity = cosine_similarity([tp_sample_.values], negative_classify_sample_.values).tolist()[0]
    change_to = None
    # for all similarity from the most equal
    for ind, sim in sorted(list(enumerate(similarity)), key=lambda x: x[1], reverse=True):
        n_sample = negative_samples.iloc[ind].to_dict()
        can_change, X_changeable = check_changeable(n_sample, tp_dic)
        if not can_change: continue
        if change_to is None: change_to = []
        change_to.append(X_changeable)
        if break_after_one_example:
            break
    return change_to


def find_change_plan(change_to, diff):
    for key in diff:
        change_to[key] = diff[key][1]
    return change_to


def play_game(X_test, y_true, y_pred, name_project, name_model=None):
    def find_negative_example(X_test, y_true, y_pred):
        negative_classify = []
        for (_, x), (_, t), p in zip(X_test.iterrows(), y_true.iteritems(), y_pred):
            if p == 0:  # find all negative instance
                negative_classify.append(x.copy().to_list())
        return pd.DataFrame(negative_classify, columns=X_test.columns)

    """
   A main function that find instance classification
   as positive and change them to the negative space
   :param X_test: x
   :param y_true: real label
   :param y_pred: predict label
   :return:
   """
    negative_classify = find_negative_example(X_test, y_true, y_pred)
    X_play = []
    new_y_true = []
    index = 0
    index_success = 0
    for (_, x), (_, t), p in zip(X_test.iterrows(), y_true.iteritems(), y_pred):
        if t == 1 and p == 1:  # find all True positive instance
            index += 1
            results = get_similarities(x, negative_classify, name_project, name_model)
            if results is not None:
                index_success += 1
                for sample in results:
                    X_play.append(list(sample.values()))
                    # update_key_rules([key for key in sample if sample[key] != dict(x)[key]])
                    # for key in sample:
                    #     counter = sample[key] - dict(x)[key]
                    #     if counter !=0:
                    #         print(f"{key} : {counter}")
                    new_y_true.append(t)
                    # print(f"change in {name_project}")
            else:  # can't move
                X_play.append(x.copy().to_list())
                new_y_true.append(t)
        else:
            X_play.append(x.copy().to_list())
            new_y_true.append(t)
    print(f"For {name_project} get {index} example change {index_success}")
    return pd.DataFrame([list(i) for i in X_play], columns=X_test.columns), new_y_true


class Move:
    def can_apply(self, x):
        return True

    def apply(self, x, key=None):
        pass

    def apply_if_can(self, x, key=None):
        if self.can_apply(x):
            self.apply(x, key)
            return True
        return False


class Unreachable(Move):
    def can_apply(self, x):
        return True

    def apply(self, x, key=None):
        if "used_added_lines+used_removed_lines" in x:  x['used_added_lines+used_removed_lines'] += 3
        if "used_added_lines-used_removed_lines" in x:  x['used_added_lines-used_removed_lines'] += 3
        if "delta_TLOC" in x:  x['delta_TLOC'] += 3
        if "delta_NL" in x:  x['delta_NL'] += 1
        if "delta_LOC" in x:  x['delta_LOC'] += 3
        if "delta_LLOC" in x:  x['delta_LLOC'] += 3
        if "delta_TLLOC" in x:  x['delta_TLLOC'] += 3
        if "delta_NOS" in x:  x['delta_NOS'] += 2
        if "delta_NLE" in x:  x['delta_NLE'] += 1
        if "delta_MethodInvocation" in x:  x['delta_MethodInvocation'] += 1
        if "delta_SwitchStatement" in x:  x['delta_SwitchStatement'] += 1
        if "delta_Literal" in x:  x['delta_Literal'] += 2
        if "delta_StatementExpression" in x:  x['delta_StatementExpression'] += 1
        if "delta_SwitchStatementCase" in x:  x['delta_SwitchStatementCase'] += 1
        if "ast_diff_condBranCaseAdd" in x:  x['ast_diff_condBranCaseAdd'] += 1
        if "ast_diff_mcAdd" in x:  x['ast_diff_mcAdd'] += 1
        if key in x:  x[key] += 1


class ChangeIfToFor(Move):
    def can_apply(self, x):
        ans = True
        if "delta_IfStatement" in x: ans = x['delta_IfStatement'] > 0
        if 'current_IfStatement' in x: ans = ans and x['current_IfStatement'] > 0
        if 'ast_diff_condBranIfAdd' in x: ans = ans and x['ast_diff_condBranIfAdd'] > 0
        if 'ast_diff_condExpExpand' in x: ans = ans and x['ast_diff_condExpExpand'] > 0
        return ans

    def apply(self, x, key=None):
        if "delta_IfStatement" in x:  x['delta_IfStatement'] -= 1
        if "delta_ForControl" in x:  x['delta_ForControl'] += 1
        if "delta_BreakStatement" in x:  x['delta_BreakStatement'] += 1
        if "delta_ForStatement" in x:  x['delta_ForStatement'] += 1
        if "ast_diff_condBranIfAdd" in x:  x['ast_diff_condBranIfAdd'] -= 1
        if "ast_diff_condExpExpand" in x:  x['ast_diff_condExpExpand'] -= 1
        if "ast_diff_loopAdd" in x:  x['ast_diff_loopAdd'] += 1
        if "delta_LOC" in x:  x['delta_LOC'] += 1
        if "delta_LLOC" in x:  x['delta_LLOC'] += 1
        if "delta_TLLOC" in x:  x['delta_TLLOC'] += 1
        if "delta_NOS" in x:  x['delta_NOS'] += 1
        if "delta_TLOC" in x:  x['delta_TLOC'] += 1
        if "used_added_lines-used_removed_lines" in x:  x['delta_IfStatement'] += 1
        if "used_added_lines-used_removed_lines" in x:  x['delta_IfStatement'] += 1


class ChangeIfToWhile(Move):
    def can_apply(self, x):
        ans = True
        if "delta_IfStatement" in x: ans = x['delta_IfStatement'] > 0
        if "ast_diff_condExpExpand" in x: ans = ans and x['ast_diff_condExpExpand'] > 0
        if "ast_diff_condBranIfAdd" in x: ans = ans and x['ast_diff_condBranIfAdd'] > 0
        if "current_IfStatement" in x: ans = ans and x['current_IfStatement'] > 0
        return ans

    def apply(self, x, key=None):
        if "delta_IfStatement" in x:  x['delta_IfStatement'] -= 1
        if "delta_BreakStatement" in x:  x['delta_BreakStatement'] += 1
        if "delta_WhileStatement" in x:  x['delta_WhileStatement'] += 1
        if "ast_diff_condBranIfAdd" in x:  x['ast_diff_condBranIfAdd'] -= 1
        if "ast_diff_condExpExpand" in x:  x['ast_diff_condExpExpand'] -= 1
        if "ast_diff_loopAdd" in x:  x['ast_diff_loopAdd'] += 1
        if "delta_LOC" in x:  x['delta_LOC'] += 1
        if "delta_LLOC" in x:  x['delta_LLOC'] -= 1
        if "delta_TLLOC" in x:  x['delta_TLLOC'] += 1
        if "delta_NOS" in x:  x['delta_NOS'] += 1
        if "delta_TLOC" in x:  x['delta_TLOC'] += 1
        if "used_added_lines-used_removed_lines" in x:  x['used_added_lines-used_removed_lines'] += 1
        if "used_added_lines+used_removed_lines" in x:  x['used_added_lines+used_removed_lines'] += 1


class ChangeMcParAdd(Move):
    def can_apply(self, x):
        if "ast_diff_mcParAdd" in x and 'ast_diff_mcAdd' in x:
            return x['ast_diff_mcAdd'] < x['mcParAdd']
        return False

    def apply(self, x, key=None):
        if "delta_MemberReference" in x:  x['delta_MemberReference'] += 1
        if "ast_diff_mcParAdd" in x:  x['ast_diff_mcParAdd'] -= 1


class mcRem(Move):
    def can_apply(self, x):
        ans = True
        if "ast_diff_mcRem" in x:  ans = x['ast_diff_mcRem'] > 0
        if 'ast_diff_mcParValChange' in x:  ans = ans and x['ast_diff_mcParValChange'] > 1
        return ans

    def apply(self, x, key=None):
        if "delta_MethodInvocation" in x:  x['delta_MethodInvocation'] += 1
        if "current_MethodInvocation" in x:  x['current_MethodInvocation'] += 1
        if "ast_diff_mcMove" in x:  x['ast_diff_mcMove'] += 1
        if "ast_diff_mcRem" in x:  x['ast_diff_mcRem'] -= 1
        if "ast_diff_mcParValChange" in x:  x['ast_diff_mcParValChange'] -= 2
        if "used_added_lines-used_removed_lines" in x:  x['used_added_lines-used_removed_lines'] += 4
        if "used_added_lines+used_removed_lines" in x:  x['used_added_lines+used_removed_lines'] += 4
        if "StatementExpression" in x:  x['StatementExpression'] += 1
        if "delta_SwitchStatement" in x:  x['delta_SwitchStatement'] += 1
        if "delta_Literal" in x:  x['delta_Literal'] += 2
        if "delta_SwitchStatementCase" in x:  x['delta_SwitchStatementCase'] += 1


class ChangeloopInitChange(Move):
    def can_apply(self, x):
        ans = True
        if "current_VariableDeclaration" in x:  ans = x['current_VariableDeclaration'] > 0
        if "delta_VariableDeclaration" in x:  ans = ans and x['delta_VariableDeclaration'] > 0
        if "ast_diff_assignExpChange" in x:  ans = ans and x['ast_diff_assignExpChange'] > 0
        if "ast_diff_loopInitChange" in x:  ans = ans and x['ast_diff_loopInitChange'] > 0
        return ans

    def apply(self, x, key=None):
        if "delta_VariableDeclaration" in x:  x['delta_VariableDeclaration'] -= 1
        if "delta_LocalVariableDeclaration" in x:  x['delta_LocalVariableDeclaration'] += 1
        if "ast_diff_assignAdd" in x:  x['ast_diff_assignAdd'] += 1
        if "ast_diff_assignRem" in x:  x['ast_diff_assignRem'] += 1
        if "ast_diff_assignExpChange" in x:  x['ast_diff_assignExpChange'] -= 1
        if "ast_diff_loopInitChange" in x:  x['ast_diff_loopInitChange'] -= 1
        if "ast_diff_objInstMod" in x:  x['ast_diff_objInstMod'] += 2
        if "ast_diff_varAdd" in x:  x['ast_diff_varAdd'] += 1
        if "delta_LOC" in x:  x['delta_LOC'] += 1
        if "delta_LLOC" in x:  x['delta_LLOC'] += 1
        if "delta_TLLOC" in x:  x['delta_TLLOC'] += 1
        if "delta_TLOC" in x:  x['delta_TLOC'] += 1
        if "used_added_lines+used_removed_lines" in x:  x['used_added_lines+used_removed_lines'] += 2
        if "used_added_lines-used_removed_lines" in x:  x['used_added_lines-used_removed_lines'] += 2

