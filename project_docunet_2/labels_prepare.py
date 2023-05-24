
def prepare_cros_re_meta_en(tokenizer):
    ents_list = ['AGE', 'AWARD', 'CITY', 'COUNTRY', 'CRIME', 'DATE', 'DISEASE', 'DISTRICT', 'EVENT', 'FACILITY',
                 'FAMILY', 'IDEOLOGY', 'LANGUAGE', 'LAW', 'LOCATION', 'MONEY', 'NATIONALITY', 'NUMBER', 'ORDINAL',
                 'ORGANIZATION', 'PENALTY', 'PERCENT', 'PERSON', 'PRODUCT', 'PROFESSION', 'RELIGION',
                 'STATE_OR_PROVINCE', 'TIME', 'WORK_OF_ART', 'ANATOMY', 'LIVB', 'ACTIVITY', 'CHEM', 'DEVICE', 'PHYS',
                 'MEDPROC', 'SCIPROC', 'LABPROC', 'BIOLOGICAL_PHEN', 'MENTALPROC', 'GENE', 'FOOD', 'INJURY_POISONING',
                 'ADMINISTRATION_ROUTE', 'HEALTH_CARE_ACTIVITY']
    tokens_start_id = 626 # unused language letters
    ents_token_alias_id = [tokenizer.convert_ids_to_tokens([tokens_start_id + i])[0] for i in range(len(ents_list))]
    ents_ids = {i[1]: i[0] for i in zip(ents_token_alias_id, ents_list)}
    rels_list = ['AFFECTS', 'AGENT', 'AGE_DIED_AT', 'AGE_IS', 'APPLIED_TO',
                 'ASSOCIATED_WITH', 'AWARDED_WITH',
                 'CAUSE_OF_DEATH', 'CONVICTED_OF', 'DATE_DEFUNCT_IN', 'DATE_FOUNDED_IN', 'DATE_OF_BIRTH',
                 'DATE_OF_CREATION', 'DATE_OF_DEATH', 'DOSE_IS', 'DURATION_OF', 'END_TIME', 'EXPENDITURE', 'FINDING_OF',
                 'FOUNDED_BY', 'HAS_ADMINISTRATION_ROUTE', 'HAS_CAUSE', 'HAS_STRENGTH', 'HEADQUARTERED_IN',
                 'IDEOLOGY_OF', 'INANIMATE_INVOLVED', 'INCOME', 'KNOWS', 'LOCATED_IN', 'MEDICAL_CONDITION', 'MEMBER_OF',
                 'MENTAL_PROCESS_OF', 'ORGANIZES', 'ORIGINS_FROM', 'OWNER_OF', 'PARENT_OF', 'PARTICIPANT_IN',
                 'PART_OF', 'PENALIZED_AS', 'PHYSIOLOGY_OF', 'PLACE_OF_BIRTH', 'PLACE_OF_DEATH', 'PLACE_RESIDES_IN',
                 'POINT_IN_TIME', 'PRICE_OF', 'PROCEDURE_PERFORMED', 'PRODUCES', 'RELATIVE', 'RELIGION_OF',
                 'SCHOOLS_ATTENDED', 'SIBLING', 'SPOUSE', 'START_TIME', 'SUBCLASS_OF', 'SUBEVENT_OF', 'SUBORDINATE_OF',
                 'TAKES_PLACE_IN', 'TO_DETECT_OR_STUDY', 'TREATED_USING', 'USED_IN', 'VALUE_IS', 'WORKPLACE',
                 'WORKS_AS']
    rel_ids = {rels_list[i]: i for i in range(len(rels_list))}
    return ents_ids, rel_ids

# DISEASE - DISO
# BIOLOGICAL_PHEN - FINDING

# {'BIOLOGICAL_PHEN_OF', 'CHEMICAL_USED', 'HAS_SYMPTOM', 'TO_DETECT', 'TO_STUDY'}
# {'ASSOCIATED_WITH', 'FINDING_OF', 'TO_DETECT_OR_STUDY', 'USED_IN'}

def prepare_cros_re_meta_ru(tokenizer):
    ents_list = ['AGE', 'AWARD', 'CITY', 'COUNTRY', 'CRIME', 'DATE', 'DISO', 'DISTRICT', 'EVENT', 'FACILITY',
                 'FAMILY', 'IDEOLOGY', 'LANGUAGE', 'LAW', 'LOCATION', 'MONEY', 'NATIONALITY', 'NUMBER', 'ORDINAL',
                 'ORGANIZATION', 'PENALTY', 'PERCENT', 'PERSON', 'PRODUCT', 'PROFESSION', 'RELIGION',
                 'STATE_OR_PROVINCE', 'TIME', 'WORK_OF_ART', 'ANATOMY', 'LIVB', 'ACTIVITY', 'CHEM', 'DEVICE', 'PHYS',
                 'MEDPROC', 'SCIPROC', 'LABPROC', 'FINDING', 'MENTALPROC', 'GENE', 'FOOD', 'INJURY_POISONING',
                 'ADMINISTRATION_ROUTE', 'HEALTH_CARE_ACTIVITY']
    tokens_start_id = 626 # unused language letters
    ents_token_alias_id = [tokenizer.convert_ids_to_tokens([tokens_start_id + i])[0] for i in range(len(ents_list))]
    ents_ids = {i[1]: i[0] for i in zip(ents_token_alias_id, ents_list)}
    rels_list = ['SUBCLASS_OF', 'MEDICAL_CONDITION', 'PART_OF', 'HAS_CAUSE', 'PHYSIOLOGY_OF',
                                 'ASSOCIATED_WITH', 'FINDING_OF', 'TREATED_USING', 'AFFECTS', 'TO_DETECT_OR_STUDY',
                                 'USED_IN', 'PROCEDURE_PERFORMED', 'VALUE_IS', 'ABBREVIATION', 'ALTERNATIVE_NAME',
                                 'ORIGINS_FROM', 'APPLIED_TO', 'MEMBER_OF', 'PARTICIPANT_IN', 'LOCATED_IN',
                                 'AGE_IS', 'TAKES_PLACE_IN', 'MENTAL_PROCESS_OF', 'PLACE_RESIDES_IN', 'DOSE_IS',
                                 'HAS_ADMINISTRATION_ROUTE', 'DURATION_OF']
    rel_ids = {rels_list[i]: i for i in range(len(rels_list))}
    return ents_ids, rel_ids

