import mysql.connector

TRIALS = [1, 2, 3]
BATCH_SIZE = [2, 4, 8, 16, 32]
TIME_STEPS = [6, 12, 24]

cnx = mysql.connector.connect(user='root', password='root',
                              host='127.0.0.1',
                              database='cec')

cursor = cnx.cursor()

file = open("load_trials.sql", "w")

for trial in TRIALS:
    for batch_size in BATCH_SIZE:
        for time_steps in TIME_STEPS:
            sql = (
                f"LOAD DATA LOCAL INFILE '/home/pablo/PycharmProjects/cec/models/trial-0{trial}/{batch_size}/{time_steps}/log.csv'"
                " INTO TABLE cec.trials"
                " FIELDS terminated by ','"
                " LINES terminated by '\n'"
                " IGNORE 1 LINES"
                " (epoch, accuracy, loss, val_accuracy, val_loss)"
                f" SET trial = {trial}, batch_size = {batch_size}, time_steps = {time_steps};\n")

            file.writelines(sql)
            # cursor.execute(sql) Oops root doesn't have permissions

file.close()

cnx.close()
