def _group_train_sequence_distance(self, evaluation_data):
    train_act_sequence_distances = [eval_dict['train_sequence_distance'] for eval_dict in evaluation_data]

    bin_cutoffs = [0, 1, 5, 10, 20, 30, 40, 50]
    bin_labels = [f'bin_{i}' for i in range(1, len(bin_cutoffs) + 1)]

    distances_series = pd.Series(train_act_sequence_distances)
    binned_distances = pd.cut(distances_series, bins=bin_cutoffs + [np.inf], labels=bin_labels, right=False,
                              include_lowest=True)

    case_bins = dict()
    for i, d in enumerate(evaluation_data):
        case_bins[f'{d["case_id"]}_{d["test_index"]}'] = binned_distances.iloc[i]

    bin_ranges = {}
    for i in range(len(bin_cutoffs)):
        lower_bound = bin_cutoffs[i]
        upper_bound = bin_cutoffs[i + 1] if i + 1 < len(bin_cutoffs) else np.inf
        bin_ranges[f'bin_{i + 1}'] = (lower_bound, upper_bound)

    for bin_label in bin_labels:
        eval_group = EvaluationGroup(
            title=f'Test instances where activity sequence distance is in {str(bin_ranges[bin_label])}',
            name=f'train_sequence_distance_{bin_label}',
            evaluation_dicts=[d for d in evaluation_data if
                              case_bins[f'{d["case_id"]}_{d["test_index"]}'] == bin_label],
            group_property_value=bin_label)
        self.evaluation_groups.append(eval_group)

    return


def _group_attribute_drift(self, evaluation_data):
    pattern = re.compile(r'attr_drift_(.*)')
    attr_drift_eval_dicts = defaultdict(list)

    for eval_dict in evaluation_data:
        for eval_property in eval_dict.keys():
            match = pattern.match(eval_property)
            if match:
                attr_drift_eval_dicts[match.group(1)].append(eval_dict)

    for attr in attr_drift_eval_dicts.keys():
        eval_group = EvaluationGroup(
            title=f'Test instances with attribute drift for {attr}',
            name=f'attr_drift_{attr}',
            evaluation_dicts=attr_drift_eval_dicts[attr],
            group_property_value=attr)
        self.evaluation_groups.append(eval_group)

    del pattern
    return


def _convert_to_hashable_dicts(self, evaluation_data):
    return [HashableDict(d) for d in evaluation_data]


def _create_evaluation_groups(self, evaluation_data):
    # TODO: Metrics for baseline proba vs naive proba for target, possibly remove my metric from thesis.

    evaluation_data = self._convert_to_hashable_dicts(evaluation_data)
    self._group_train_sequence_distance(evaluation_data)
    self._group_attribute_drift(evaluation_data)
    return