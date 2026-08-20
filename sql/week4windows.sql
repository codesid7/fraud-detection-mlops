-- Quest 1 -- This is the same question as problem #11 in the SQL Chapter of Ace the Data Science Interview!

-- Assume you are given the table below on Uber transactions made by users. Write a query to obtain the third transaction of every user. Output the user id, spend and transaction date.

-- transactions Table:
-- Column Name	Type
-- user_id	integer
-- spend	decimal
-- transaction_date	timestamp
-- transactions Example Input:
-- user_id	spend	transaction_date
-- 111	100.50	01/08/2022 12:00:00
-- 111	55.00	01/10/2022 12:00:00
-- 121	36.00	01/18/2022 12:00:00
-- 145	24.99	01/26/2022 12:00:00
-- 111	89.60	02/05/2022 12:00:00
-- Example Output:
-- user_id	spend	transaction_date
-- 111	89.60	02/05/2022 12:00:00
-- The dataset you are querying against may have different input & output - this is just an example!

--solution
SELECT user_id, spend, transaction_date
FROM (SELECT user_id, spend, transaction_date,
ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY transaction_date ASC) AS transaction_number
FROM transactions) AS T
WHERE transaction_number = 3;

--Question 2
-- Imagine you're an HR analyst at a tech company tasked with analyzing employee salaries. Your manager is keen on understanding the pay distribution and asks you to determine the second highest salary among all employees.

-- It's possible that multiple employees may share the same second highest salary. In case of duplicate, display the salary only once.

-- employee Schema:
-- column_name	type	description
-- employee_id	integer	The unique ID of the employee.
-- name	string	The name of the employee.
-- salary	integer	The salary of the employee.
-- department_id	integer	The department ID of the employee.
-- manager_id	integer	The manager ID of the employee.
-- employee Example Input:
-- employee_id	name	salary	department_id	manager_id
-- 1	Emma Thompson	3800	1	6
-- 2	Daniel Rodriguez	2230	1	7
-- 3	Olivia Smith	2000	1	8
-- Example Output:
-- second_highest_salary
-- 2230
-- The output represents the second highest salary among all employees. In this case, the second highest salary is $2,230.

---SOLUTION
SELECT MAX(salary) AS second_highest_salary
FROM employee
WHERE salary < (
    SELECT MAX(salary)
    FROM employee
);

