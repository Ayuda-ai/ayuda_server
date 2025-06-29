from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.course import Course
from app.core.dependencies import admin_required
from app.services.course_service import embed_and_index_courses
import pandas as pd
import uuid

router = APIRouter()

@router.post("/parse_courses")
def parse_courses(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user = Depends(admin_required),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Parse and import courses from uploaded Excel or CSV file with background embedding generation.
    
    This endpoint allows administrators to bulk import courses from spreadsheet files.
    The function processes the uploaded file, validates course data, and stores courses
    in the database. After successful import, it triggers background embedding generation
    and indexing in Pinecone for course matching functionality.
    
    Supported file formats:
    - application/vnd.openxmlformats-officedocument.spreadsheetml.sheet (.xlsx)
    - text/csv (.csv)
    
    Required columns in the file:
    - course_id: Unique identifier for the course
    - course_name: Name of the course
    - course_description: Detailed description of the course
    - major: Academic major/field the course belongs to
    
    Optional columns:
    - prerequisite_1, prerequisite_2, prerequisite_3: Course prerequisites
    - domain_1, domain_2: Course domains/categories
    - skills_associated: Comma-separated list of skills covered
    
    The function automatically creates a combined_text field that concatenates
    all course information for embedding generation.
    
    Args:
        file (UploadFile): Excel or CSV file containing course data
        db (Session): Database session dependency
        current_user: Authenticated admin user (from admin_required dependency)
        background_tasks (BackgroundTasks): FastAPI background tasks for async processing
        
    Returns:
        dict: Import results:
            - message: Success message with number of courses processed
            
    Raises:
        HTTPException: If file type is unsupported (status_code=400), file is empty (status_code=400),
                      validation fails (status_code=422), or processing fails (status_code=500)
    """
    # Check file content type
    if file.content_type not in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "text/csv"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only .xlsx and .csv files are accepted."
        )

    try:
        if file.filename.endswith(".xlsx"):
            df = pd.read_excel(file.file)
        elif file.filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file extension. Only .xlsx and .csv allowed."
            )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read uploaded file. Ensure it's a valid .xlsx or .csv. Error: {str(e)}"
        )

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    inserted_courses = []

    for _, row in df.iterrows():
        try:
            # Extract fields with safety checks
            course_id = str(row.get("course_id", "")).strip()
            course_name = str(row.get("course_name", "")).strip()
            course_description = str(row.get("course_description", "")).strip()

            prerequisites = [
                str(row.get("prerequisite_1", "")).strip(),
                str(row.get("prerequisite_2", "")).strip(),
                str(row.get("prerequisite_3", "")).strip()
            ]
            prerequisites = [p for p in prerequisites if p] or None

            major = str(row.get("major", "")).strip()

            domains = [
                str(row.get("domain_1", "")).strip(),
                str(row.get("domain_2", "")).strip()
            ]
            domains = [d for d in domains if d] or None

            skills = str(row.get("skills_associated", "")).strip()
            skills_list = [skill.strip() for skill in skills.split(',')] if skills else None

            # Validate required fields
            if not course_id or not course_name or not course_description or not major:
                raise ValueError(f"Missing mandatory fields for a course at row {_+2}")  # +2 considering header and 0-based index

            # Create combined text
            combined_text_parts = [
                f"Course Name: {course_name}",
                f"Description: {course_description}",
                f"Prerequisites: {', '.join(prerequisites) if prerequisites else 'None'}",
                f"Major: {major}",
                f"Domains: {', '.join(domains) if domains else 'None'}",
                f"Skills: {', '.join(skills_list) if skills_list else 'None'}"
            ]
            combined_text = ". ".join(combined_text_parts)

            existing_course = db.query(Course).filter_by(course_id=course_id).first()
            course_id_to_use = existing_course.id if existing_course else uuid.uuid4()
            # Create Course object
            course = Course(
                id=course_id_to_use,
                course_id=course_id,
                course_name=course_name,
                course_description=course_description,
                prerequisites=prerequisites,
                major=major,
                domains=domains,
                skills_associated=skills_list,
                combined_text=combined_text
            )
            # inserted_courses.append(course)
            db.merge(course);

        except ValueError as ve:
            raise HTTPException(status_code=422, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error processing course at row {_+2}: {str(e)}")

    try:
        # db.add_all(inserted_courses)
        db.commit()
    except Exception as db_error:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(db_error)}")

    # 🔁 Kick off the background embedding/indexing task
    background_tasks.add_task(embed_and_index_courses)

    return {"message": f"Successfully inserted {len(inserted_courses)} courses."}
