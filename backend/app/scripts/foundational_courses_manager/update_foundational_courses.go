package main

import (
	"bufio"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	_ "github.com/lib/pq"
)

// FoundationalCourse represents a foundational course configuration
type FoundationalCourse struct {
	Major                 string   `json:"major"`
	FoundationalCourseIDs []string `json:"foundational_course_ids"`
}

// loadEnv loads environment variables from .env file
func loadEnv() map[string]string {
	env := make(map[string]string)

	// Try to find .env file in project root
	envPath := filepath.Join("..", "..", "..", ".env")
	if _, err := os.Stat(envPath); os.IsNotExist(err) {
		// Try current directory
		envPath = ".env"
	}

	file, err := os.Open(envPath)
	if err != nil {
		log.Printf("Warning: Could not open .env file: %v", err)
		return env
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		parts := strings.SplitN(line, "=", 2)
		if len(parts) == 2 {
			key := strings.TrimSpace(parts[0])
			value := strings.TrimSpace(parts[1])
			env[key] = value
		}
	}

	return env
}

// updateFoundationalCourses updates the foundational_courses table
func updateFoundationalCourses(jsonFile string) error {
	// Load environment variables
	env := loadEnv()

	// Get database credentials from environment
	dbHost := env["DATABASE_HOST"]
	dbPort := env["DATABASE_PORT"]
	dbUser := env["POSTGRES_USER"]
	dbPassword := env["POSTGRES_PASSWORD"]
	dbName := env["POSTGRES_DB"]

	if dbHost == "" || dbPort == "" || dbUser == "" || dbPassword == "" || dbName == "" {
		return fmt.Errorf("missing database credentials in .env file")
	}

	// Connect to database
	dsn := fmt.Sprintf("host=%s port=%s user=%s password=%s dbname=%s sslmode=disable",
		dbHost, dbPort, dbUser, dbPassword, dbName)

	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return fmt.Errorf("failed to connect to database: %v", err)
	}
	defer db.Close()

	// Test connection
	if err := db.Ping(); err != nil {
		return fmt.Errorf("failed to ping database: %v", err)
	}

	// Read JSON file
	data, err := os.ReadFile(jsonFile)
	if err != nil {
		return fmt.Errorf("failed to read JSON file: %v", err)
	}

	// Parse JSON
	var courses []FoundationalCourse
	if err := json.Unmarshal(data, &courses); err != nil {
		return fmt.Errorf("failed to parse JSON: %v", err)
	}

	// Start transaction
	tx, err := db.Begin()
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %v", err)
	}
	defer tx.Rollback()

	// Clear existing data
	if _, err := tx.Exec("DELETE FROM foundational_courses"); err != nil {
		return fmt.Errorf("failed to clear existing data: %v", err)
	}

	// Insert new data with PostgreSQL UUID generation
	stmt, err := tx.Prepare(`
		INSERT INTO foundational_courses (id, major, foundational_course_ids, created_at, updated_at)
		VALUES (gen_random_uuid(), $1, $2, NOW(), NOW())
	`)
	if err != nil {
		return fmt.Errorf("failed to prepare statement: %v", err)
	}
	defer stmt.Close()

	// Insert each course
	for _, course := range courses {
		courseIDsStr := "{" + strings.Join(course.FoundationalCourseIDs, ",") + "}"

		if _, err := stmt.Exec(course.Major, courseIDsStr); err != nil {
			return fmt.Errorf("failed to insert course for major %s: %v", course.Major, err)
		}
		fmt.Printf("✅ Added foundational courses for %s: %v\n", course.Major, course.FoundationalCourseIDs)
	}

	// Commit transaction
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %v", err)
	}

	fmt.Printf("✅ Successfully updated %d foundational course configurations\n", len(courses))
	return nil
}

func main() {
	if len(os.Args) != 2 {
		fmt.Println("Usage: go run update_foundational_courses.go <json_file>")
		fmt.Println("Example: go run update_foundational_courses.go foundational_courses.json")
		os.Exit(1)
	}

	jsonFile := os.Args[1]

	if err := updateFoundationalCourses(jsonFile); err != nil {
		log.Fatalf("Error: %v", err)
	}
}
