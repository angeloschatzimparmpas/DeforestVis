# first line: 433
    @memory.cache
    def estimator(n_estimators, learning_rate):
        # initialize model
        print('modelsCompNow!')
        n_estimators = int(n_estimators)
        model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=RANDOM_SEED)
        model.fit(X_train, predictions)
        # set in cross-validation
        #result = cross_validate(model, X_train, predictions, cv=crossValidation, scoring='accuracy')
        # result is mean of test_score
        return model.score(X_train, predictions)
