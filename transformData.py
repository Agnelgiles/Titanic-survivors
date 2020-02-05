import numpy as np


def create_age_bin(age, start, end):
    if start < age <= end:
        return 1
    else:
        return 0


def transform_age(data):
    data['Age'] = data['Age'].fillna(data['Age'].mean())
    data['age_bin1'] = data['Age'].apply(create_age_bin, args=(0, 20))
    data['age_bin2'] = data['Age'].apply(create_age_bin, args=(20, 40))
    data['age_bin3'] = data['Age'].apply(create_age_bin, args=(40, 60))
    data['age_bin4'] = data['Age'].apply(create_age_bin, args=(60, 80))
    return data.drop(['Age'], axis=1)


def convert_male_sex(sex):
    if sex.lower() == 'male':
        return 1
    else:
        return 0


def convert_female_sex(sex):
    if sex.lower() == 'female':
        return 1
    else:
        return 0


def transform_sex(data):
    data['is_male'] = data['Sex'].apply(convert_male_sex)
    data['is_female'] = data['Sex'].apply(convert_female_sex)
    return data.drop(['Sex'], axis=1)


def create_embarked(emarked, station):
    if emarked == station:
        return 1
    else:
        return 0


def transform_embarked(data):
    data['embarked_s'] = data['Embarked'].apply(create_embarked, args='S')
    data['embarked_c'] = data['Embarked'].apply(create_embarked, args='C')
    data['embarked_q'] = data['Embarked'].apply(create_embarked, args='Q')
    return data.drop(['Embarked'], axis=1)


def transform_fare(data):
    data['Fare'] = np.log((1 + data['Fare']))
    return data


def create_data_frame(data):
    data = transform_age(data)
    data = transform_sex(data)
    data = transform_embarked(data)
    data = transform_fare(data)
    data['Pclass'] = data['Pclass'].round()
    data = data.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1)
    data = data.fillna(0)
    return data
