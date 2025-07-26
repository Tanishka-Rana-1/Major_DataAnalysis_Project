CREATE DATABASE linkedin_jobs;
USE linkedin_jobs;
SHOW VARIABLES LIKE 'secure_file_priv';
LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/jobs/cleaned_salaries.csv'
INTO TABLE salaries
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES;
SELECT * FROM salaries LIMIT 5;
SELECT COUNT(*) FROM salaries;
LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/jobs/cleaned_job_industries.csv'
INTO TABLE job_industries
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES;
SELECT * FROM job_industries LIMIT 5;
SELECT COUNT(*) FROM job_industries;
LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/jobs/cleaned_job_skills.csv'
INTO TABLE job_skills
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES;
SELECT * FROM job_skills LIMIT 5;
SELECT COUNT(*) FROM job_skills;
ALTER TABLE companies MODIFY COLUMN zip_code VARCHAR(20);
LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/companies/cleaned_companies.csv'
INTO TABLE companies
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(company_id, name, description, company_size, state, country, city, @zip_code, address, url)
SET zip_code = LEFT(@zip_code, 20);
ALTER TABLE companies MODIFY COLUMN city VARCHAR(255);
SELECT * FROM companies LIMIT 5;
SELECT COUNT(*) FROM companies;
DROP TABLE IF EXISTS company_specialities;

CREATE TABLE company_specialities (
    company_id INT DEFAULT NULL,
    speciality_name TEXT,
    description TEXT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

ALTER TABLE company_specialities MODIFY COLUMN speciality_name VARCHAR(8000);

SHOW CREATE TABLE company_specialities;
LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/companies/cleaned_company_specialities.csv'
INTO TABLE company_specialities
CHARACTER SET utf8mb4
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(company_id, speciality_name, @dummy)
SET description = NULL;
SELECT * FROM company_specialities LIMIT 5;
SELECT COUNT(*) FROM company_specialities;

DROP TABLE IF EXISTS employee_counts;
CREATE TABLE employee_counts (
    company_id INT,
    employee_count INT,
    follower_count INT,
    time_recorded INT
);
LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/companies/cleaned_employee_counts.csv'
INTO TABLE employee_counts
CHARACTER SET utf8mb4
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(company_id, employee_count, follower_count, time_recorded);
SELECT * FROM employee_counts LIMIT 5;
SELECT COUNT(*) FROM employee_counts;

DROP TABLE IF EXISTS industries;
CREATE TABLE industries (
    industry_id INT,
    industry_name VARCHAR(255)
);
LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/mappings/cleaned_industries.csv'
INTO TABLE industries
CHARACTER SET utf8mb4
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(industry_id, industry_name);
SELECT * FROM industries LIMIT 5;
SELECT COUNT(*) FROM industries;

DROP TABLE IF EXISTS skills;
CREATE TABLE skills (
    skill_abr VARCHAR(10),
    skill_name VARCHAR(50)
);
LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/mappings/cleaned_skills.csv'
INTO TABLE skills
CHARACTER SET utf8mb4
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(skill_abr, skill_name);
SELECT * FROM skills LIMIT 5;
SELECT COUNT(*) FROM skills;

-- 1. Joins companies, employee_counts, and company_specialities to provide company names, employee counts, and specialities
SELECT 
    c.name AS company_name,
    c.company_size,
    e.employee_count,
    e.follower_count,
    cs.speciality_name
FROM companies c
LEFT JOIN employee_counts e ON c.company_id = e.company_id
LEFT JOIN company_specialities cs ON c.company_id = cs.company_id;

-- 2. Joins salaries, job_industries, and industries to analyze salaries across industries.

SELECT 
    i.industry_name,
    s.avg_salary,
    s.job_id
FROM salaries s
LEFT JOIN job_industries ji ON s.job_id = ji.job_id
LEFT JOIN industries i ON ji.industry_id = i.industry_id;

-- 3. Company Location and Employee Summary

SELECT 
    c.name AS company_name,
    e.employee_count,
    e.follower_count
FROM companies c
LEFT JOIN employee_counts e ON c.company_id = e.company_id;




