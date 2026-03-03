from django.db import models

class Document(models.Model):
    file_name = models.CharField(max_length=200)
    chunk_text = models.TextField()
    embedding = models.JSONField(default=list, blank=True)

    def __str__(self):
        return self.file_name