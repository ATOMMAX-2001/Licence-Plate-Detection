import sqlite3
connection=sqlite3.connect('hackathon.db')
print("sucessful")
connection.execute('''create table theftVehicle(id int auto_increment,username varchar(255),color varchar(255),plate varchar(255),blackORstolen varchar(255) ,primary key(id));''')
print ("table sucessfully")

connection.execute('''INSERT INTO theftVehicle(username,color,plate,blackORstolen) VALUES('ramakrishna','blue','DZ17YXR','stolen');''')
connection.execute('''INSERT INTO theftVehicle(username,color,plate,blackORstolen) VALUES('abilash','black','DL7CQ1939','blacklisted');''')
connection.execute('''INSERT INTO theftVehicle(username,color,plate,blackORstolen) VALUES('nirmal','blue','FT856VD','stolen');''')
connection.execute('''INSERT INTO theftVehicle(username,color,plate,blackORstolen) VALUES('balaji','white','TN21BZ0768','blacklisted');''')
connection.execute('''INSERT INTO theftVehicle(username,color,plate,blackORstolen) VALUES('arun','brown','MH01TMP8145','stolen');''')
connection.execute('''INSERT INTO theftVehicle(username,color,plate,blackORstolen) VALUES('rakesh','red','TN09BV6196','blacklisted');''')
connection.execute('''INSERT INTO theftVehicle(username,color,plate,blackORstolen) VALUES('hari','red','WB24AK0333','stolen');''')
connection.execute('''INSERT INTO theftVehicle(username,color,plate,blackORstolen) VALUES('peter','black','TN09BU1357','blacklisted');''')
connection.execute('''INSERT INTO theftVehicle(username,color,plate,blackORstolen) VALUES('krish','white','TN22DK3510','stolen');''')
connection.execute('''INSERT INTO theftVehicle(username,color,plate,blackORstolen) VALUES('karthik','violet','RJ14CE5678','blacklisted');''')
connection.execute('''INSERT INTO theftVehicle(username,color,plate,blackORstolen) VALUES('atommax','brown','MH46N4832','stolen');''')












print("row is inserted")
